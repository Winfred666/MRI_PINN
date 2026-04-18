import gc
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import url2pathname

import torch
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import MLFlowLogger

from utils.config_loader import Train_Config
from utils.mlflow_logging import log_file_artifact, log_text_artifact


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
    """Keep one MLflow run across multiple Lightning Trainer.fit() calls."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._auto_hparams_logged = False
        self._defer_finalize = True
        self._finalized = False
        self._pending_finalize_status = None
        self.active_phase = None
        self.metric_step_offset = 0

    def log_hyperparams(self, params):
        if self._auto_hparams_logged:
            return
        super().log_hyperparams(params)
        self._auto_hparams_logged = True

    def log_metrics(self, metrics, step=None):
        offset_step = None if step is None else int(step) + int(self.metric_step_offset)
        super().log_metrics(metrics, step=offset_step)

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

    def set_active_phase(self, phase: str | None) -> None:
        phase_name = str(phase or "").strip()
        self.active_phase = phase_name or None

    def absolute_step(self, step: int | None = None) -> int:
        return int(self.metric_step_offset) + int(step or 0)

    def advance_step_offset(self, phase_steps: int) -> None:
        self.metric_step_offset += max(0, int(phase_steps))


def _set_mlflow_tag(logger: PhaseSafeMLFlowLogger | None, key: str, value: Any) -> None:
    if logger is None:
        return
    tag_key = re.sub(r"[^0-9A-Za-z_.\- /:]", "_", str(key).strip())
    if not tag_key:
        raise ValueError("MLflow tag key resolved to an empty string after sanitization.")
    logger.experiment.set_tag(logger.run_id, tag_key, str(value))


def _logger_artifact_uri(logger: Any) -> str:
    """Resolve the active run artifact URI from a configured MLflow logger."""
    if logger is None:
        raise ValueError("MLflow logger is required to resolve the dynamic artifact URI.")

    experiment = getattr(logger, "experiment", None)
    get_run = getattr(experiment, "get_run", None)
    run_id_raw = getattr(logger, "run_id", None)
    run_id = str(run_id_raw).strip() if run_id_raw is not None else ""
    if not callable(get_run) or not run_id:
        raise ValueError("logger must expose `experiment.get_run` and a non-empty `run_id`.")

    run = get_run(run_id)
    run_info = getattr(run, "info", None)
    artifact_uri = str(getattr(run_info, "artifact_uri", "")).strip() if run_info is not None else ""
    if not artifact_uri:
        raise ValueError(f"MLflow run '{run_id}' does not expose a non-empty artifact URI.")
    return artifact_uri


def artifact_uri_to_local_path(artifact_uri: str) -> Path:
    """Convert a local path or file:// artifact URI into an absolute local path."""
    normalized_uri = str(artifact_uri).strip()
    if not normalized_uri:
        raise ValueError("artifact_uri must be a non-empty local path or file URI.")

    parsed = urlparse(normalized_uri)
    if parsed.scheme not in {"", "file"}:
        raise ValueError(
            f"Only local file-backed MLflow artifact URIs are supported, got '{artifact_uri}'."
        )

    if parsed.scheme == "file":
        if parsed.netloc not in {"", "localhost"}:
            raise ValueError(
                f"Only local file-backed MLflow artifact URIs are supported, got '{artifact_uri}'."
            )
        local_path = url2pathname(parsed.path or "")
    else:
        local_path = normalized_uri

    if not local_path:
        raise ValueError("artifact_uri resolved to an empty local path.")
    return Path(local_path).expanduser().resolve()


def _resolve_run_output_dir(train_config: Train_Config, logger: PhaseSafeMLFlowLogger | None) -> Path:
    if logger is None:
        output_dir = Path(train_config.result_folder).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    artifact_dir = artifact_uri_to_local_path(_logger_artifact_uri(logger))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    placeholder_dir = Path(train_config.result_folder).expanduser().resolve()
    train_config.result_folder = str(artifact_dir)
    if placeholder_dir != artifact_dir and placeholder_dir.is_dir():
        try:
            next(placeholder_dir.iterdir())
        except StopIteration:
            placeholder_dir.rmdir()
    return artifact_dir


def _build_checkpoint_callback(phase_ckpt_dir: Path, ckpt_every_n_epochs: int) -> ModelCheckpoint:
    return ModelCheckpoint(
        monitor="hp_metric",
        mode="min",
        save_top_k=2,
        save_last=True,
        save_on_exception=True,
        auto_insert_metric_name=False,
        filename="epoch{epoch:05d}-step{step:08d}-hp{hp_metric:.6f}",
        dirpath=str(phase_ckpt_dir),
        every_n_epochs=ckpt_every_n_epochs,
    )


def _phase_snapshot_path(phase_ckpt_dir: Path, model_type: str, phase: str) -> Path:
    snapshot_dir = phase_ckpt_dir / "phase_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    return snapshot_dir / f"{model_type}pinn_{phase}.pth"


def _build_trainer(
    train_config: Train_Config,
    logger: PhaseSafeMLFlowLogger | None,
    checkpoint_callback: ModelCheckpoint,
    max_epochs: int,
) -> Trainer:
    return Trainer(
        max_epochs=int(max_epochs),
        default_root_dir=train_config.result_folder,
        reload_dataloaders_every_n_epochs=train_config.reload_dataloaders_every_n_epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=train_config.ckpt_save_val_interval,
        num_sanity_val_steps=(2 if train_config.enable_validation else 0),
        limit_val_batches=(1.0 if train_config.enable_validation else 0),
        callbacks=[checkpoint_callback],
        logger=(logger if logger is not None else False),
    )


def _resolve_checkpoint_phase(checkpoint_phase: Any, train_phase_order: list[str]) -> str:
    phase_name = str(checkpoint_phase or "").strip()
    if not phase_name:
        raise ValueError("Checkpoint does not contain a non-empty train_phase.")
    if phase_name in train_phase_order:
        return phase_name

    checkpoint_parts = {part.strip() for part in phase_name.split("+") if part.strip()}
    ranked_matches: list[tuple[int, int, str]] = []
    for phase in train_phase_order:
        phase_parts = {part.strip() for part in phase.split("+") if part.strip()}
        if phase_parts and phase_parts.issubset(checkpoint_parts):
            ranked_matches.append((len(phase_parts), len(phase), phase))

    if ranked_matches:
        ranked_matches.sort(reverse=True)
        return ranked_matches[0][2]

    raise ValueError(
        f"Checkpoint phase '{phase_name}' is not present in config phases {train_phase_order}."
    )


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
            run_name=train_config.run_name,
            log_model=False,
        )
    else:
        raise ValueError(
            f"Unsupported logging backend '{train_config.logging_backend}'. Supported: mlflow | none"
        )

    run_output_dir = _resolve_run_output_dir(train_config, logger)

    if logger is not None:
        _set_mlflow_tag(logger, "run.name", train_config.run_name)
        _set_mlflow_tag(logger, "run.output_dir", str(run_output_dir))
        log_file_artifact(logger=logger, local_path=train_config.yaml_path, artifact_path="run")

    train_phase_order = list(train_config.phases.keys())
    train_phase_togo = train_phase_order
    resume_epoch = 0
    strict_resume_ckpt_path = None

    if train_config.continue_training:
        checkpoint = torch.load(train_config.ckpt_path, map_location="cpu")
        if "state_dict" not in checkpoint:
            raise ValueError(f"Checkpoint '{train_config.ckpt_path}' does not contain a Lightning state_dict.")
        checkpoint_phase = _resolve_checkpoint_phase(checkpoint.get("train_phase"), train_phase_order)
        train_phase_togo = train_phase_order[train_phase_order.index(checkpoint_phase):]
        strict_resume_ckpt_path = train_config.ckpt_path
        resume_epoch = int(checkpoint.get("epoch", 0))
        print(f"Continue training from phase {checkpoint_phase} with epoch {resume_epoch}")

    last_model = None
    current_phase = None

    try:
        for phase in train_phase_togo:
            current_phase = phase
            cfg = train_config.phases[phase]
            lr = cfg.get("learning_rate", 1e-3)
            max_epochs = int(cfg.get("max_epochs", 1000))
            batch_size = cfg.get("batch_size", 222_000)
            phase_ckpt_dir = run_output_dir / "checkpoints" / phase
            phase_ckpt_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_callback = _build_checkpoint_callback(
                phase_ckpt_dir=phase_ckpt_dir,
                ckpt_every_n_epochs=train_config.ckpt_save_val_interval,
            )

            phase_resume_ckpt_path = strict_resume_ckpt_path if phase == train_phase_togo[0] else None
            trainer_max_epochs = resume_epoch + max_epochs if phase_resume_ckpt_path else max_epochs
            trainer = _build_trainer(
                train_config=train_config,
                logger=logger,
                checkpoint_callback=checkpoint_callback,
                max_epochs=trainer_max_epochs,
            )

            pinn_model, datamodule = trainer_getter(phase, main_net, cfg)
            if pinn_model is None or datamodule is None:
                continue

            pinn_model.hparams.learning_rate = lr
            pinn_model.train_phase = phase

            pinn_model.set_enable_rbar(train_config.enable_rbar)
            if train_config.enable_rbar:
                datamodule.set_RBA_resample_model(pinn_model)

            datamodule.batch_size = batch_size

            print(
                f"PHASE_START phase={phase} lr={lr} batch_size={batch_size} "
                f"phase_max_epochs={max_epochs} trainer_max_epochs={trainer_max_epochs}"
            )

            if logger is not None:
                logger.set_active_phase(phase)
                _set_mlflow_tag(logger, "phase.current", phase)
                _set_mlflow_tag(logger, f"phase.{phase}.status", "running")
                _set_mlflow_tag(logger, f"phase.{phase}.checkpoint_dir", str(phase_ckpt_dir))
                log_text_artifact(
                    logger=logger,
                    text=phase,
                    text_key=f"phase/start/{phase}",
                    step=logger.absolute_step(),
                )
                log_text_artifact(
                    logger=logger,
                    text=yaml.safe_dump(cfg, sort_keys=False),
                    text_key=f"phase/config/{phase}",
                    step=logger.absolute_step(),
                )

            trainer.fit(
                pinn_model,
                datamodule=datamodule,
                ckpt_path=phase_resume_ckpt_path,
            )
            strict_resume_ckpt_path = None

            phase_snapshot_path = _phase_snapshot_path(
                phase_ckpt_dir=phase_ckpt_dir,
                model_type=train_config.model_type,
                phase=phase,
            )
            trainer.save_checkpoint(str(phase_snapshot_path))

            if logger is not None:
                _set_mlflow_tag(logger, f"phase.{phase}.status", "finished")
                _set_mlflow_tag(logger, "phase.current", phase)
                _set_mlflow_tag(logger, f"checkpoint.{phase}.phase_snapshot", str(phase_snapshot_path))
                if checkpoint_callback.best_model_path:
                    _set_mlflow_tag(logger, f"checkpoint.{phase}.best_model_path", checkpoint_callback.best_model_path)
                if checkpoint_callback.last_model_path:
                    _set_mlflow_tag(logger, f"checkpoint.{phase}.last_model_path", checkpoint_callback.last_model_path)
                logger.advance_step_offset(trainer.global_step)

            print(
                f"PHASE_DONE phase={phase} current_epoch={trainer.current_epoch} "
                f"global_step={trainer.global_step} ckpt={phase_snapshot_path}"
            )

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            last_model = pinn_model

        if logger is not None:
            logger.finalize_run("success")
        return last_model
    except Exception:
        if logger is not None and current_phase is not None:
            _set_mlflow_tag(logger, f"phase.{current_phase}.status", "failed")
            _set_mlflow_tag(logger, "phase.current", current_phase)
            logger.finalize_run("failed")
        raise
