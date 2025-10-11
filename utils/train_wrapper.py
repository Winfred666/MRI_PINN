from lightning.pytorch import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import torch
from utils.config_loader import Train_Config

def train_all_phases(main_net, trainer_getter, train_config: Train_Config):
    logger = TensorBoardLogger("tb_logs", name="seqtrain_ADPINN",
                               version=train_config.result_folder.split('/')[-1])
    # save every N epochs
    checkpoint_callback_best = ModelCheckpoint(
        filename="pinn-{epoch:04d}",
        monitor='hp_metric',
        save_top_k=2,
        every_n_epochs=train_config.ckpt_save_val_interval
    )
    checkpoint_callback_latest = ModelCheckpoint(
        filename="pinn-latest-{epoch:04d}",
        save_top_k=0,
        save_last=True,
        every_n_epochs=train_config.ckpt_save_val_interval
    )
    
    trainer = Trainer(
        reload_dataloaders_every_n_epochs=train_config.reload_dataloaders_every_n_epochs,
        log_every_n_steps=5,
        check_val_every_n_epoch=train_config.ckpt_save_val_interval,
        callbacks=[checkpoint_callback_best, checkpoint_callback_latest],
        logger=logger
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

    for phase in train_phase_togo:
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

        trainer.logger.experiment.add_text("phase/start", phase, global_step=trainer.global_step)
        
        if continue_training_from_ckpt:
            pinn_model.load_state_dict(checkpoint['state_dict'], strict=continue_training_from_ckpt_strict)
            continue_training_from_ckpt = False # only use for the first phase
            continue_training_from_ckpt_strict = False # only use for the first phase

        # WARNING: here direcly continue training from checkpoint if specified and last_model is None
        trainer.fit(
            pinn_model, 
            datamodule=datamodule,
            ckpt_path=train_config.ckpt_path if continue_training_from_ckpt_strict else None
        )

        ckpt_name = f"{train_config.result_folder}/adpinn_{phase}.pth"
        trainer.save_checkpoint(ckpt_name)
        
        last_model = pinn_model
    
    return last_model
