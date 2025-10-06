from modules.c_net import C_Net, MLP
from modules.rba_resample_trainer import Net_RBAResample
import torch
import numpy as np

class CNet_Init(Net_RBAResample):
    def on_save_checkpoint(self, checkpoint):
        checkpoint['train_phase'] = 'init_c'

    def __init__(self, c_net: C_Net, num_train_points,
                 learning_rate=1e-3, rba_learning_rate=0.1, rba_memory=0.999,
                 enable_rbar=True):
        super().__init__([("init_c_data", 1.0)], num_train_points,
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
    def on_save_checkpoint(self, checkpoint):
        checkpoint['train_phase'] = 'init_c_denoise'

    def __init__(self, c_net: C_Net, num_train_points,
                 learning_rate=1e-3,
                 rba_learning_rate=0.1, rba_memory=0.999,
                 input_dim=4, freq_nums=(8,8,8,0), gamma_space=1.0,
                 warmup_freeze_c_epochs: int = 0,
                 sigma_reg_weight: float = 0.001,
                 enable_rbar=True):
        # Use its own RBA entry name to distinguish logs if desired
        super().__init__([("init_c_denoise_data", 1.0)], num_train_points,
                         rba_learning_rate, rba_memory, enable_rbar=enable_rbar)

        self.save_hyperparameters(ignore=['c_net'])
        self.c_net = c_net
        self.warmup_freeze_c_epochs = warmup_freeze_c_epochs
        self.sigma_reg_weight = sigma_reg_weight

        self.noise_mlp = MLP(
            input_dim=c_net.c_mlp.input_dim,
            output_dim=1,
            hidden_layers=5,  # from Table D.8
            hidden_features=66
        )
        self.sigma_0 = 0.01  # minimum noise level

    def forward(self, X_train):
        c_clean_pred = self.c_net(X_train)
        if self.c_net.positional_encoding:
            txyz_freq = self.c_net.c_pos_encoder(X_train)
        else:
            txyz_freq = X_train
        predicted_sigma = 10.0 * torch.sigmoid(self.noise_mlp(txyz_freq)) + self.sigma_0
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
        Xt, X_train_indice, c_observed = batch
        X_full = torch.cat(Xt, dim=1)
        c_clean_pred, sigma_pred = self(X_full)

        # Negative log likelihood per-point
        errp2 = (c_observed - c_clean_pred) ** 2
        eps = 1e-8
        log_term = torch.log((sigma_pred / self.sigma_0) ** 2 + eps) / 2
        quad_term = errp2 / (2.0 * (sigma_pred ** 2 + eps))
        nll_loss = log_term + quad_term

        if self.sigma_reg_weight > 0:
            target = errp2 + eps
            reg = (torch.log(sigma_pred ** 2 + eps) - torch.log(target)) ** 2
            nll_loss = nll_loss + self.sigma_reg_weight * reg

        if batch_idx == 0:
            self.log('train_c_denoise_sigma', sigma_pred.mean())
            self.log('train_c_denoise_data_loss', errp2.mean())

        # RBA optimization
        super().training_step(self.c_net, X_train_indice, nll_loss, Xt, batch_idx)

    def validation_step(self, batch, batch_idx):
        Xt, _, c_observed = batch
        X_full = torch.cat(Xt, dim=1)
        c_clean_pred, _ = self(X_full)
        loss_data = ((c_observed - c_clean_pred) ** 2).mean()
        if batch_idx == 0:
            self.log('val_data_loss', loss_data)
            c_vis_list = self.c_net.draw_concentration_slices()
            self.logger.experiment.add_image('val_C_compare', c_vis_list, self.current_epoch, dataformats='WH')
        return loss_data

    # def configure_optimizers(self):
    #     return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
