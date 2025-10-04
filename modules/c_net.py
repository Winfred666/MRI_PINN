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
    def __init__(self, c_net : C_Net, input_dim=4, lr=1e-3, freq_nums=(8,8,8,0), gamma_space=1.0):
        """
        Args:
            input_dim: 4 (for t, x, y, z)
            c_net: concentration network to be pre-train
            data: only for validation visualization
            lr: learning rate
            positional_encoding: only setting to noise network
            freq_nums: (fx, fy, fz, ft) frequencies, also only setting to noise network
            gamma_space: spatial scaling for positional encoding, only setting to noise network
        """
        super().__init__()
        self.save_hyperparameters()

        # Positional encoding setup (shared for both heads)
        self.positional_encoding = c_net.positional_encoding
        # frequency of noise could be higher than that of clean signal
        if self.positional_encoding:
            num_freq_space = freq_nums[:3]
            num_freq_time = freq_nums[3]
            self.pos_encoder = PositionalEncoding_GeoTime(
                num_freq_space,
                num_freq_time,
                include_input=True,
                gamma_space=gamma_space,
            )
            mlp_input_dim = 4 + int(np.array(num_freq_space).sum()) * 2 + int(num_freq_time) * 2
        else:
            self.pos_encoder = None
            mlp_input_dim = input_dim
        self.c_net = c_net

        # Network for the noise standard deviation (sigma)
        # Output is 1 because we predict a single sigma value per point
        self.nn_noise = MLP(
            input_dim=mlp_input_dim,
            output_dim=1,
            hidden_layers=5,
            hidden_features=66 # From Table D.8
        )

    def forward(self, txyz_coords):
        """
        A forward pass returns both the clean signal and the predicted noise.
        """
        
        # Predict the clean concentration
        c_clean_pred = self.c_net(txyz_coords)

        # Optionally encode inputs once and share for both heads
        if self.pos_encoder is not None:
            txyz_freq = self.pos_encoder(txyz_coords)
        else:
            txyz_freq = txyz_coords

        # Predict the noise standard deviation using the paper's formulation
        # Equation A.12: sigma = 10 * sigmoid(NN_noise) + sigma_0
        sigma_0 = 0.01
        
        # We use sigmoid to bound the output of nn_noise between 0 and 1
        predicted_sigma = 10.0 * torch.sigmoid(self.nn_noise(txyz_freq)) + sigma_0
        
        return c_clean_pred, predicted_sigma

    def nll_loss(self, c_observed, c_clean_pred, sigma_pred):
        """
        Calculates the Negative Log-Likelihood loss as per Equation A.14.
        """
        # The paper uses a simplified NLL for a Gaussian distribution
        # Term 1: log(sigma^2)
        log_variance = torch.log(sigma_pred.pow(2))
        
        # Term 2: (observed - predicted)^2 / (2 * sigma^2)
        squared_error = (c_observed - c_clean_pred).pow(2)
        error_term = squared_error / (2 * sigma_pred.pow(2))

        # We are ignoring the constant C_o from the paper as it doesn't affect optimization
        loss = log_variance + error_term
        
        return loss.mean()

    def training_step(self, batch, batch_idx):
        # The batch should contain the coordinates and the observed (noisy) concentration
        txyz_coords, c_observed = batch
        
        # Get predictions from the model
        c_clean_pred, sigma_pred = self(txyz_coords)

        # Calculate the NLL loss
        loss = self.nll_loss(c_observed, c_clean_pred, sigma_pred)

        # Log metrics for monitoring
        self.log("train_loss", loss, prog_bar=True)
        self.log("predicted_sigma_mean", sigma_pred.mean())
        
        return loss

    def validation_step(self, batch, batch_idx):
        txyz_coords, c_observed = batch
        c_clean_pred, sigma_pred = self(txyz_coords)
        loss = self.nll_loss(c_observed, c_clean_pred, sigma_pred)
        if batch_idx == 0:
            self.log("val_loss", loss)
            if self.current_epoch % 50 == 0:
                c_vis_list = self.c_net.draw_concentration_slices()
                self.logger.experiment.add_image('val_C_compare', c_vis_list, self.current_epoch, dataformats='WH')
                # also draw noise's

    def configure_optimizers(self):
        # We optimize both networks together using a single optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer