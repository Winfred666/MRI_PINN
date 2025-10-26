from modules.c_net import C_Net, MLP
from modules.rba_resample_trainer import Net_RBAResample
import torch
import numpy as np

class CNet_Init(Net_RBAResample):
    train_phase = "init_c_data"
    def __init__(self, c_net: C_Net, num_train_points,
                 learning_rate=1e-3, rba_learning_rate=0.1, rba_memory=0.999,
                 enable_rbar=True):
        super().__init__([(CNet_Init.train_phase, 1.0)], num_train_points,
                         rba_learning_rate, rba_memory, enable_rbar=enable_rbar)

        self.save_hyperparameters(ignore=['c_net'])
        self.c_net = c_net

    def forward(self, X_train):
        return self.c_net(X_train)

    def training_step(self, batch, batch_idx):
        Xt, X_train_indice, c_observed = batch
        
        # 1. Forward pass
        C_pred = self(torch.cat(Xt, dim=1))

        # 2. Calculate per-point weighted MSE loss (for RBA update)
        pointwise_loss = (C_pred - c_observed) ** 2
        # Log the mean loss for monitoring(already done in RBA base class)
        # 3,4,5. Perform TDRBA-weighted optimization step and update RBA
        super().training_step(self.c_net, X_train_indice, pointwise_loss, Xt, batch_idx)
    

    def validation_step(self, batch, batch_idx):
        Xt, _, c_observed = batch
        mod_out = self(torch.cat(Xt, dim=1))
        # if mod_out is tuple, select the first element
        if isinstance(mod_out, tuple):
            c_clean_pred = mod_out[0]
        else:
            c_clean_pred = mod_out
        loss_data = ((c_observed - c_clean_pred) ** 2).mean()

        if batch_idx == 0:
            self.log('val_data_loss', loss_data)
            c_vis_list = self.c_net.draw_concentration_slices()
            self.logger.experiment.add_image('val_C_compare', c_vis_list, self.current_epoch, dataformats='WH')
        return loss_data
    


class CNet_DenoiseInit(Net_RBAResample):

    train_phase = "init_c_denoise_data"

    def __init__(self, c_net: C_Net, num_train_points,
                 learning_rate=1e-3,
                 rba_learning_rate=0.1, rba_memory=0.999,
                 warmup_freeze_c_epochs: int = 0,
                 enable_rbar=True):
        # Use its own RBA entry name to distinguish logs if desired
        super().__init__([(CNet_DenoiseInit.train_phase, 1.0)], num_train_points,
                         rba_learning_rate, rba_memory, enable_rbar=enable_rbar)

        self.save_hyperparameters(ignore=['c_net'])
        self.c_net = c_net
        self.warmup_freeze_c_epochs = warmup_freeze_c_epochs

    def forward(self, X_train):
        c_clean_pred = self.c_net(X_train)
        predicted_sigma = self.c_net.sigma_forward(X_train)
        return c_clean_pred, predicted_sigma

    def on_train_start(self):
        if self.warmup_freeze_c_epochs > 0:
            for p in self.c_net.parameters():
                p.requires_grad = False

    def on_train_epoch_start(self):
        if (self.warmup_freeze_c_epochs > 0 and
                self.current_epoch == self.warmup_freeze_c_epochs):
            for p in self.c_net.parameters():
                p.requires_grad = True

    def training_step(self, batch, batch_idx):
        Xt, X_train_indice, _ = batch
        nll_loss, sigma_pred, errp2 = self.c_net.calculate_denoise_loss(batch)

        if batch_idx == 0:
            self.log('train_c_denoise_sigma', sigma_pred.mean())
            self.log('train_c_denoise_data_loss', errp2.mean())

        # RBA optimization
        super().training_step(self.c_net, X_train_indice, nll_loss, Xt, batch_idx)

    def validation_step(self, batch, batch_idx):
        Xt, _, c_observed = batch
        X_full = torch.cat(Xt, dim=1)
        c_clean_pred = self.c_net(X_full)
        loss_data = ((c_observed - c_clean_pred) ** 2).mean()
        if batch_idx == 0:
            self.log('val_data_loss', loss_data)
            c_vis_list = self.c_net.draw_concentration_slices()
            self.logger.experiment.add_image('val_C_compare', c_vis_list, self.current_epoch, dataformats='WH')
        return loss_data

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
