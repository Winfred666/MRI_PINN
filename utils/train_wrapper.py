import os
import gc
from urllib.parse import urlparse

import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from utils.config_loader import Train_Config
from utils.mlflow_logging import log_text_artifact


def _configure_mlflow_proxy_bypass(tracking_uri: str):
    tracking_uri = str(tracking_uri).strip()
    if not tracking_uri:
        return

    host = ""
    parsed = urlparse(tracking_uri)
    if parsed.hostname:
        host = parsed.hostname

    no_proxy_items = []
    for key in ("NO_PROXY", "no_proxy"):
        existing = str(os.environ.get(key, "")).strip()
        if existing:
            no_proxy_items.extend([item.strip() for item in existing.split(",") if item.strip()])

    no_proxy_items.extend(["localhost", "127.0.0.1", "::1"])
    if host:
        no_proxy_items.append(host)

    dedup_no_proxy = ",".join(dict.fromkeys(no_proxy_items))
    os.environ["NO_PROXY"] = dedup_no_proxy
    os.environ["no_proxy"] = dedup_no_proxy


class PhaseSafeMLFlowLogger(MLFlowLogger):
    """Avoid MLflow param collisions when Lightning re-runs fit() for each phase.

    Lightning calls logger.log_hyperparams() at every Trainer.fit() invocation.
    This repo trains multiple phases within a single MLflow run, so phase-specific
    model hyperparameters like learning rate and RBA names legitimately change.
    MLflow params are immutable, so we only let Lightning perform its automatic
    hyperparameter logging once per run and rely on explicit phase artifacts for
    per-phase details.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_hparams_logged = False
        self._defer_finalize = True
        self._finalized = False
        self._pending_finalize_status = None

    def log_hyperparams(self, params):
        if self._auto_hparams_logged:
            return
        super().log_hyperparams(params)
        self._auto_hparams_logged = True

    def finalize(self, status: str = "success") -> None:
        if self._finalized:
            return
        if self._defer_finalize:
            self._pending_finalize_status = status
            return
        super().finalize(status)
        self._finalized = True

    def finalize_run(self, status: str | None = None) -> None:
        if self._finalized:
            return
        resolved_status = status or self._pending_finalize_status or "success"
        self._defer_finalize = False
        try:
            super().finalize(resolved_status)
            self._finalized = True
        finally:
            self._defer_finalize = True


def train_all_phases(main_net, trainer_getter, train_config: Train_Config):
    logger = None
    if train_config.logging_backend in {"none", "disabled", "false"}:
        logger = None
    elif train_config.logging_backend == "mlflow":
        _configure_mlflow_proxy_bypass(train_config.mlflow_tracking_uri)
        os.environ["MLFLOW_TRACKING_URI"] = train_config.mlflow_tracking_uri
        logger = PhaseSafeMLFlowLogger(
            experiment_name=train_config.mlflow_experiment,
            tracking_uri=train_config.mlflow_tracking_uri,
            run_name=train_config.result_folder.split('/')[-1],
            log_model=train_config.mlflow_log_model,
        )
    else:
        raise ValueError(
            f"Unsupported logging backend '{train_config.logging_backend}'. Supported: mlflow | none"
        )

    # save every N epochs
    checkpoint_callback_best = ModelCheckpoint(
        filename="pinn-{epoch}-{hp_metric:.6f}",
        monitor='hp_metric',
        save_top_k=2,
    )

    checkpoint_callback_latest = ModelCheckpoint(
        filename="pinn-latest-{epoch}",
        monitor='epoch',
        save_top_k=1,
        every_n_epochs=train_config.ckpt_save_val_interval,
    )
    
    trainer = Trainer(
        reload_dataloaders_every_n_epochs=train_config.reload_dataloaders_every_n_epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=train_config.ckpt_save_val_interval,
        num_sanity_val_steps=(2 if train_config.enable_validation else 0),
        limit_val_batches=(1.0 if train_config.enable_validation else 0),
        callbacks=[checkpoint_callback_best, checkpoint_callback_latest],
        logger=(logger if logger is not None else False),
    )
    
    last_model = None
    train_phase_togo = list(train_config.phases.keys())
    continue_training_from_ckpt = train_config.continue_training
    total_epochs = 0

    continue_training_from_ckpt_strict = False
    
    if continue_training_from_ckpt:
        checkpoint = torch.load(train_config.ckpt_path, map_location="cpu")
        # Check if the checkpoint is a Lightning checkpoint and if the phase matches
        if "state_dict" in checkpoint and checkpoint.get("train_phase") == train_phase_togo[0]:
            continue_training_from_ckpt_strict = True
            last_epoch = checkpoint["epoch"]
            total_epochs = last_epoch
            print(f"Continue training from phase {checkpoint['train_phase']} with epoch {last_epoch}")

    current_phase = None
    try:
        for phase in train_phase_togo:
            current_phase = phase
            cfg = train_config.phases[phase]
            lr = cfg.get("learning_rate", 1e-3)
            max_epochs = cfg.get("max_epochs", 1000)
            batch_size = cfg.get("batch_size", 222_000)
            
            # get correct model and datamodule for the phase
            pinn_model, datamodule = trainer_getter(phase, main_net, cfg)
            if pinn_model is None or datamodule is None:
                continue
            pinn_model.hparams.learning_rate = lr # Set learning rate for the phase
            pinn_model.train_phase = phase # Tag the model with the current phase for checkpointing

            pinn_model.set_enable_rbar(train_config.enable_rbar)
            if train_config.enable_rbar:
                datamodule.set_RBA_resample_model(pinn_model)

            datamodule.batch_size = batch_size
            
            total_epochs += max_epochs
            trainer.fit_loop.max_epochs = total_epochs

            print(
                f"PHASE_START phase={phase} lr={lr} batch_size={batch_size} "
                f"phase_max_epochs={max_epochs} cumulative_max_epochs={total_epochs}"
            )

            if logger is not None:
                logger.experiment.set_tag(logger.run_id, "phase/current", phase)
                logger.experiment.set_tag(logger.run_id, f"phase/{phase}/status", "running")
                log_text_artifact(
                    logger=logger,
                    text=phase,
                    text_key="phase/start",
                    step=trainer.global_step,
                )
                log_text_artifact(
                    logger=logger,
                    text=yaml.safe_dump(cfg, sort_keys=False),
                    text_key=f"phase/config/{phase}",
                    step=trainer.global_step,
                )
            
            if continue_training_from_ckpt:
                print("load from checkpoint...")
                loading_report = pinn_model.load_state_dict(
                    checkpoint['state_dict'],
                    strict=continue_training_from_ckpt_strict,
                )
                
                print(f"Checkpoint loading report:")
                if loading_report.missing_keys:
                    print(f"  - Missing keys in model: {loading_report.missing_keys}")
                if loading_report.unexpected_keys:
                    print(f"  - Unexpected keys in checkpoint: {loading_report.unexpected_keys}")
                if not loading_report.missing_keys and not loading_report.unexpected_keys:
                    print("  - All keys matched successfully.")

                continue_training_from_ckpt = False # only use for the first phase

            # WARNING: here direcly continue training from checkpoint if specified and last_model is None
            trainer.fit(
                pinn_model, 
                datamodule=datamodule,
                ckpt_path=train_config.ckpt_path if continue_training_from_ckpt_strict else None
            )

            if continue_training_from_ckpt_strict:
                continue_training_from_ckpt_strict = False # already used.

            ckpt_name = f"{train_config.result_folder}/{train_config.model_type}pinn_{phase}.pth"
            trainer.save_checkpoint(ckpt_name)
            if logger is not None:
                logger.experiment.set_tag(logger.run_id, f"phase/{phase}/status", "finished")
                logger.experiment.set_tag(logger.run_id, "phase/current", phase)
            print(
                f"PHASE_DONE phase={phase} current_epoch={trainer.current_epoch} "
                f"global_step={trainer.global_step} ckpt={ckpt_name}"
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            last_model = pinn_model

        if logger is not None and hasattr(logger, "finalize_run"):
            logger.finalize_run("success")
        return last_model
    except Exception:
        if logger is not None and current_phase is not None:
            logger.experiment.set_tag(logger.run_id, f"phase/{current_phase}/status", "failed")
            logger.experiment.set_tag(logger.run_id, "phase/current", current_phase)
        if logger is not None and hasattr(logger, "finalize_run"):
            logger.finalize_run("failed")
        raise
