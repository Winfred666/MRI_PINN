from modules.ad_net import AD_Net
import numpy as np
import torch
import torch.nn as nn
from modules.rba_resample_trainer import Net_RBAResample


class ADPINN_Base(Net_RBAResample):
    """
    Base trainer that:
      - Integrates with Net_RBAResample for adaptive resampling
      - Provides a unified validation_step (expects concentration observations)
    """
    def __init__(self,
                 ad_net: AD_Net,
                 num_train_points: int,
                 rba_name_weight_list,
                 learning_rate=1e-3,
                 rba_learning_rate=0.1,
                 rba_memory=0.999,
                 distribution_shapness=2.0,
                 base_probability=0.5,
                 enable_rbar=True):
        super().__init__(rba_name_weight_list,
                         num_train_points,
                         rba_learning_rate,
                         rba_memory,
                         distribution_shapness,
                         base_probability,
                         enable_rbar)
        self.save_hyperparameters(ignore=['ad_net'])
        self.ad_net = ad_net
        self.c_net = ad_net.c_net
        self.v_net = ad_net.v_net
        self.learning_rate = learning_rate
        self.L2_loss = nn.MSELoss()

    # def configure_optimizers(self):
    #     params = [p for p in self.parameters() if p.requires_grad]
    #     return torch.optim.AdamW(params, lr=self.learning_rate)

    # already setting check_val_every_n_epoch
    def validation_step(self, batch, batch_idx):
        """
        Expects batch like (Xt, X_train_indice, c_observed)
        Xt: tuple (X, t); X:(N,3), t:(N,1)
        """
        Xt, _, c_observed = batch
        x_full = torch.cat(Xt, dim=1)
        with torch.no_grad():
            c_pred = self.c_net(x_full)
        val_loss = ((c_pred - c_observed) ** 2).mean()

        if batch_idx == 0:
            self.log('val_data_loss', val_loss)
            c_vis = self.c_net.draw_concentration_slices()
            self.logger.experiment.add_image('val_C_compare', c_vis, self.current_epoch, dataformats='WH')
            rgb_img, vx, vy, vz = self.v_net.draw_velocity_volume()
            self.logger.experiment.add_image('val_v_quiver', rgb_img, self.current_epoch, dataformats='HWC')

            # log velocity histogram
            flat_v = np.sqrt(vx**2 + vy**2 + vz**2).flatten()
            self.logger.experiment.add_histogram('val_v_hist', flat_v, self.current_epoch)
            
        return val_loss


class ADPINN_InitV(ADPINN_Base):
    """
    Velocity initialization using observed velocity samples.
    Batch format: (Xt, X_train_indice, v_observed)
      Xt = (X, t) with shapes (N,3),(N,1)
      v_observed: (N,3)
    RBA list: one component ('init_v_data', 1.0)
    """
    train_phase = "init_v_data"
    def __init__(self,
                 ad_net: AD_Net,
                 num_train_points,
                 learning_rate=1e-3,
                 rba_learning_rate=0.1,
                 rba_memory=0.999,
                 enable_rbar=True):
        super().__init__(ad_net,
                         num_train_points,
                         rba_name_weight_list=[("init_v_data", 1.0)],
                         learning_rate=learning_rate,
                         rba_learning_rate=rba_learning_rate,
                         rba_memory=rba_memory,
                         enable_rbar=enable_rbar)

    def training_step(self, batch, batch_idx):
        X, X_train_indice, v_observed = batch
        vx, vy, vz = self.v_net(X)
        v_pred = torch.cat([vx, vy, vz], dim=1)  # (N,3)

        # Per-point squared error summed over components
        pointwise_loss = ((v_pred - v_observed) ** 2).mean(dim=1, keepdim=True)

        # Pass to RBA (wrap in list)
        super().training_step(None, X_train_indice, [pointwise_loss], None, batch_idx)


class ADPINN_PDE_V(ADPINN_Base):
    """
    PDE-based refinement of velocity (c frozen).
    Batch: (Xt, X_train_indice, c_dummy) where c_dummy can be zeros (unused).
    RBA list: ('pde_v_residual', 1.0)
    """
    train_phase = "ad_pde_v"
    def __init__(self,
                 ad_net: AD_Net,
                 num_train_points,
                 learning_rate=1e-3,
                 rba_learning_rate=0.1,
                 rba_memory=0.999,
                 enable_rbar=True):
        super().__init__(ad_net,
                         num_train_points,
                         rba_name_weight_list=[(ADPINN_PDE_V.train_phase, 1.0)],
                         learning_rate=learning_rate,
                         rba_learning_rate=rba_learning_rate,
                         rba_memory=rba_memory,
                         enable_rbar=enable_rbar)
        # Freeze c_net
        for p in self.c_net.parameters():
            p.requires_grad = False
        self.c_net.eval()

    def training_step(self, batch, batch_idx):
        Xt, X_train_indice, _ = batch
        # PDE residual at points
        pde_residual = self.ad_net.pde_residual(Xt)  # (N,1), retain graph
        pointwise_loss = (pde_residual ** 2)

        if batch_idx == 0:
            self.log('pde_v_D', self.ad_net.D.item())

        super().training_step(self.c_net, X_train_indice, [pointwise_loss], Xt, batch_idx)


class ADPINN_Joint(ADPINN_Base):
    """
    Joint training of c_net and v_net with data + PDE losses.
    Batch: (Xt, X_train_indice, c_observed)
    RBA list: [('joint_data', 1.0), ('joint_pde', 10.0)]
    Weight 10.0 is applied inside Net_RBAResample for sampling & optimization.
    """

    train_phase = "joint_data+joint_ad_pde"

    def __init__(self,
                 ad_net: AD_Net,
                 num_train_points,
                 learning_rate=1e-3,
                 rba_learning_rate=0.1,
                 rba_memory=0.999,
                 enable_rbar=True):
        super().__init__(ad_net,
                         num_train_points,
                         rba_name_weight_list=[("joint_data", 1.0),
                                               ("joint_ad_pde", 10.0)],
                         learning_rate=learning_rate,
                         rba_learning_rate=rba_learning_rate,
                         rba_memory=rba_memory,
                         enable_rbar=enable_rbar)
        # Unfreeze c_net
        for p in self.c_net.parameters():
            p.requires_grad = True
        self.c_net.train()

    def training_step(self, batch, batch_idx):
        Xt, X_train_indice, c_observed = batch
        x_full = torch.cat(Xt, dim=1)

        # Data loss (apply same weighting strategy as CNet_Init)
        c_pred = self.c_net(x_full)
        data_weight = 1.0 + 0.09 * c_observed
        data_pointwise = data_weight * (c_pred - c_observed) ** 2  # (N,1)

        # PDE residual loss per point
        pde_residual = self.ad_net.pde_residual(Xt)  # (N,1)
        pde_pointwise = pde_residual ** 2

        if batch_idx == 0:
            self.log('joint_D', self.ad_net.D.item())

        # Two-loss list (must align with rba_name_weight_list order)
        super().training_step(self.c_net, X_train_indice,
                              [data_pointwise, pde_pointwise],
                              Xt, batch_idx)