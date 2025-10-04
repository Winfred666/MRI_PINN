from modules.data_module import CharacteristicDomain
import torch
import torch.nn as nn
import lightning as L
import numpy as np

from modules.positional_encoding import PositionalEncoding_GeoTime
from utils.visualize import visualize_prediction_vs_groundtruth

# Containing basic concentration network, 
# and denoised concentration network (2 sub network)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, hidden_features):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_features), nn.SiLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            # WARNING: use SiLU instead of tanh to avoid vanishing gradient
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_features, output_dim))
        self.net = nn.Sequential(*layers)

        # initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class C_Net(nn.Module):
    # data is only for validation visualization
    def __init__(
        self,
        c_layers,
        data,mask, char_domain:CharacteristicDomain,C_star,
        positional_encoding=True,
        freq_nums = (8,8,8,0),
        gamma_space=1.0,
    ):
        super().__init__()
        self.c_layers = c_layers
        self.positional_encoding = positional_encoding

        if positional_encoding:
            num_freq_space = freq_nums[:3]
            num_freq_time = freq_nums[3]
            # always include input as non periodic representation.
            c_pos_encoder = PositionalEncoding_GeoTime(
                num_freq_space,
                num_freq_time,
                include_input=True,
                gamma_space=gamma_space,
            )
            # update input layer size
            c_layers[0] = 4 + int(num_freq_space.sum()) * 2 + int(num_freq_time) * 2

        self.c_net = MLP(
            input_dim=c_layers[0],
            output_dim=c_layers[-1],
            hidden_layers=len(c_layers) - 2,
            hidden_features=c_layers[1],
        )

        self.c_net = nn.Sequential()
        if positional_encoding:
            self.c_net.add_module("c_positional_encoding", c_pos_encoder)
        
        self.c_net.add_module("c_mlp", MLP(
            input_dim=c_layers[0],
            output_dim=c_layers[-1],
            hidden_layers=len(c_layers) - 2,
            hidden_features=c_layers[1],
        ))

        self.domain_shape = char_domain.domain_shape

        self.val_slice_z = [self.domain_shape[2]//2 - 4, self.domain_shape[2]//2, self.domain_shape[2]//2 + 4]
        base_t = self.domain_shape[3]//4 * 2
        self.val_slice_t = [base_t - 6, base_t - 3 ,base_t, base_t + 3]
        self.val_slice_4d = char_domain.get_characteristic_geotimedomain(slice_zindex = self.val_slice_z, 
                                                                       slice_tindex = self.val_slice_t)
        Z,T = np.meshgrid(self.val_slice_z, self.val_slice_t, indexing='ij')
        self.val_slice_gt_c = data[:, :, Z, T] / C_star # simulate the dataset process

        self.mask = mask


    def forward(self, x):
        return self.c_net(x)

    def draw_concentration_slices(self,):
        with torch.no_grad():
            vol_disp_all = self.c_net(self.val_slice_4d).cpu().numpy().reshape(
                self.domain_shape[0], self.domain_shape[1], len(self.val_slice_z), len(self.val_slice_t))
            # self.logger.experiment.add_image('val_C_slice', vol_disp, self.current_epoch, dataformats='HW')
            # scale of vol_disp will done inside visualize function
            c_vis_list = []
            for i in range(len(self.val_slice_z)):
                for j in range(len(self.val_slice_t)):
                    slice_gt_c = self.val_slice_gt_c[:, :, i, j]
                    vol_disp = vol_disp_all[:, :, i, j]
                    vol_disp *= self.mask[:,:,self.val_slice_z[i]]  # mask out of brain region
                    c_vis_list.append(visualize_prediction_vs_groundtruth(vol_disp, slice_gt_c))
            # stack all images horizontally (along H direction)
            return np.hstack(c_vis_list)


class Denoising_C_Pretrainer(L.LightningModule):
    def __init__(self, c_net: C_Net, input_dim=4, lr=1e-3, freq_nums=(8,8,8,0), gamma_space=1.0,
                 correct_nll: bool = True,
                 detach_sigma_for_c: bool = False,
                 warmup_freeze_c_epochs: int = 50,
                 sigma_reg_weight: float = 0.01):
        super().__init__()
        self.save_hyperparameters()
        self.positional_encoding = c_net.positional_encoding
        if self.positional_encoding:
            num_freq_space = freq_nums[:3]
            num_freq_time = freq_nums[3]
            self.pos_encoder = PositionalEncoding_GeoTime(
                num_freq_space, num_freq_time,
                include_input=True,
                gamma_space=gamma_space,
            )
            mlp_input_dim = 4 + int(np.array(num_freq_space).sum()) * 2 + int(num_freq_time) * 2
        else:
            self.pos_encoder = None
            mlp_input_dim = input_dim
        self.c_net = c_net

        self.nn_noise = MLP(
            input_dim=mlp_input_dim,
            output_dim=1,
            hidden_layers=5, # from Table D.8
            hidden_features=66
        )

    def forward(self, txyz_coords):
        c_clean_pred = self.c_net(txyz_coords)
        if self.pos_encoder is not None:
            txyz_freq = self.pos_encoder(txyz_coords)
        else:
            txyz_freq = txyz_coords
        sigma_0 = 0.01
        predicted_sigma = 10.0 * torch.sigmoid(self.nn_noise(txyz_freq)) + sigma_0
        return c_clean_pred, predicted_sigma

    def nll_loss(self, c_observed, c_pred, sigma_pred):
        """
        Gaussian NLL (up to constant): log σ + (err^2)/(2 σ^2)
        If correct_nll=False fallback to previous (log σ^2 + err^2/(2 σ^2)).
        """
        err = c_observed - c_pred
        eps = 1e-8
        if self.hparams.correct_nll:
            log_term = torch.log(sigma_pred + eps)           # log σ
            quad_term = (err.pow(2)) / (2.0 * (sigma_pred.pow(2) + eps))
            base = log_term + quad_term
        else:
            log_term = torch.log(sigma_pred.pow(2) + eps)    # log σ^2 (old)
            quad_term = (err.pow(2)) / (2.0 * (sigma_pred.pow(2) + eps))
            base = log_term + quad_term

        # Optional regularizer: encourage sigma^2 ≈ err^2 (optimal for true NLL)
        if self.hparams.sigma_reg_weight > 0:
            target = err.pow(2) + eps
            reg = (torch.log(sigma_pred.pow(2) + eps) - torch.log(target)).pow(2)
            base = base + self.hparams.sigma_reg_weight * reg

        return base.mean()

    def on_train_start(self):
        # Optionally freeze c_net for warmup
        if self.hparams.warmup_freeze_c_epochs > 0:
            for p in self.c_net.parameters():
                p.requires_grad = False

    def on_train_epoch_start(self):
        # Unfreeze after warmup
        if (self.hparams.warmup_freeze_c_epochs > 0 and
                self.current_epoch == self.hparams.warmup_freeze_c_epochs):
            for p in self.c_net.parameters():
                p.requires_grad = True

    def training_step(self, batch, batch_idx):
        txyz_coords, c_observed = batch
        c_clean_pred, sigma_pred = self(txyz_coords)

        if self.hparams.detach_sigma_for_c:
            # Two-part loss: NLL for sigma net, MSE (weighted by stopped sigma) for c_net
            # 1. sigma branch
            loss_sigma = self.nll_loss(c_observed.detach(), c_clean_pred.detach(), sigma_pred)
            # 2. c branch
            eps = 1e-8
            sigma_det = sigma_pred.detach()
            mse_weighted = ((c_observed - c_clean_pred).pow(2) / (sigma_det.pow(2) + eps)).mean()
            loss = loss_sigma + mse_weighted
            self.log("train_loss_sigma", loss_sigma, prog_bar=False)
            self.log("train_loss_c_weighted", mse_weighted, prog_bar=False)
        else:
            loss = self.nll_loss(c_observed, c_clean_pred, sigma_pred)

        self.log("train_loss", loss, prog_bar=True)
        self.log("sigma_mean", sigma_pred.mean())
        self.log("sigma_min", sigma_pred.min())
        self.log("sigma_max", sigma_pred.max())
        return loss

    def validation_step(self, batch, batch_idx):
        txyz_coords, c_observed = batch
        c_clean_pred, sigma_pred = self(txyz_coords)
        loss = self.nll_loss(c_observed, c_clean_pred, sigma_pred)
        if batch_idx == 0:
            self.log("val_data_loss", loss, prog_bar=True)
            self.log("val_sigma_mean", sigma_pred.mean())
            if self.current_epoch % 50 == 0:
                c_vis_list = self.c_net.draw_concentration_slices()
                # 2D image: use 'HW'
                self.logger.experiment.add_image('val_C_compare', c_vis_list, self.current_epoch, dataformats='HW')
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)