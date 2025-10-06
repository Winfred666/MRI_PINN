import torch
import torch.nn as nn
from modules.rba_resample_trainer import Net_RBAResample
from modules.dc_net import AD_DC_Net


class DCPINN_Base(Net_RBAResample):
    """Base class for Darcy-based PINNs wrapping adaptive resampling.
    No validation_step here (can be added similarly to ADPINN_Base if needed).
    """
    def __init__(self,
                 ad_dc_net: AD_DC_Net,
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
        self.save_hyperparameters(ignore=['ad_dc_net'])
        self.ad_dc_net = ad_dc_net
        self.c_net = ad_dc_net.c_net
        self.v_dc_net = ad_dc_net.v_dc_net
        self.learning_rate = learning_rate
        self.L2_loss = nn.MSELoss()


class DCPINN_InitK(DCPINN_Base):
    """Permeability network initialization using observed velocities.
    Batch: (X, X_train_indice, v_observed) where:
      X: (N,3), spatial points
      v_observed: (N,3)
    RBA entry: ('init_k_data', 1.0)
    Only K_Net parameters are optimized; freeze P_Net & c_net.
    """
    def on_save_checkpoint(self, checkpoint):
        checkpoint['train_phase'] = 'init_k'

    def __init__(self,
                 ad_dc_net: AD_DC_Net,
                 num_train_points,
                 learning_rate=1e-3,
                 rba_learning_rate=0.1,
                 rba_memory=0.999,
                 enable_rbar=True):
        super().__init__(ad_dc_net,
                         num_train_points,
                         rba_name_weight_list=[("init_k_data", 1.0)],
                         learning_rate=learning_rate,
                         rba_learning_rate=rba_learning_rate,
                         rba_memory=rba_memory,
                         enable_rbar=enable_rbar)
        # Freeze unrelated params
        for p in self.c_net.parameters():
            p.requires_grad = False
        for p in self.v_dc_net.p_net.parameters():
            p.requires_grad = False
        # Ensure K params require grad
        for p in self.v_dc_net.k_net.parameters():
            p.requires_grad = True

    def training_step(self, batch, batch_idx):
        X, X_train_indice, v_observed = batch
        v_pred = self.v_dc_net(X)  # (N,3)
        pointwise_loss = ((v_pred - v_observed) ** 2).mean(dim=1, keepdim=True)
        if batch_idx == 0:
            self.log('train_init_k_data_loss', pointwise_loss.mean())
        # No time-dependent scaling → c_net=None
        super().training_step(None, X_train_indice, [pointwise_loss], None, batch_idx)


class DCPINN_ADPDE_P(DCPINN_Base):
    """Optimize pressure network via advection-diffusion residual (+ optional divergence residual).
    Batch: (Xt, X_train_indice, c_dummy) – c_dummy unused.
    RBA entries: [('pde_p_residual', 1.0)] or plus ('div_v_residual', div_weight) if incompressible.
    """
    def on_save_checkpoint(self, checkpoint):
        checkpoint['train_phase'] = 'pde_p'

    def __init__(self,
                 ad_dc_net: AD_DC_Net,
                 num_train_points,
                 incompressible=False,
                 div_weight=1.0,
                 learning_rate=1e-3,
                 rba_learning_rate=0.1,
                 rba_memory=0.999,
                 enable_rbar=True):
        rba_list = [("pde_p_residual", 1.0)]
        self.incompressible = incompressible
        if incompressible:
            rba_list.append(("div_v_residual", div_weight))
        super().__init__(ad_dc_net,
                         num_train_points,
                         rba_name_weight_list=rba_list,
                         learning_rate=learning_rate,
                         rba_learning_rate=rba_learning_rate,
                         rba_memory=rba_memory,
                         enable_rbar=enable_rbar)
        # Freeze c & K
        for p in self.c_net.parameters():
            p.requires_grad = False
        for p in self.v_dc_net.k_net.parameters():
            p.requires_grad = False
        # Enable p_net params
        for p in self.v_dc_net.p_net.parameters():
            p.requires_grad = True
        self.c_net.eval()

    def training_step(self, batch, batch_idx):
        Xt, X_train_indice, _ = batch
        pde_residual = self.ad_dc_net.pde_residual(Xt)  # (N,1)
        pde_pointwise = pde_residual ** 2
        loss_list = [pde_pointwise]
        if self.incompressible:
            X, _t = Xt
            div_v = self.v_dc_net.incompressible_residual(X)  # (N,1)
            div_pointwise = div_v ** 2
            loss_list.append(div_pointwise)
        if batch_idx == 0:
            self.log('pde_p_D', self.ad_dc_net.D.item())
        super().training_step(self.c_net, X_train_indice, loss_list, Xt, batch_idx)


class DCPINN_Joint(DCPINN_Base):
    """Joint optimization of c_net, k_net, p_net with data + PDE + optional divergence losses.
    Batch: (Xt, X_train_indice, c_observed)
    Default weights: data=1.0, pde=10.0, div=1.0 (if incompressible)
    """
    def on_save_checkpoint(self, checkpoint):
        checkpoint['train_phase'] = 'joint'

    def __init__(self,
                 ad_dc_net: AD_DC_Net,
                 num_train_points,
                 incompressible=False,
                 data_weight=1.0,
                 pde_weight=10.0,
                 div_weight=1.0,
                 learning_rate=1e-3,
                 rba_learning_rate=0.1,
                 rba_memory=0.999,
                 enable_rbar=True):
        rba_list = [("joint_data", data_weight), ("joint_pde", pde_weight)]
        self.incompressible = incompressible
        if incompressible:
            rba_list.append(("joint_div", div_weight))
        super().__init__(ad_dc_net,
                         num_train_points,
                         rba_name_weight_list=rba_list,
                         learning_rate=learning_rate,
                         rba_learning_rate=rba_learning_rate,
                         rba_memory=rba_memory,
                         enable_rbar=enable_rbar)
        # Unfreeze all
        for p in self.c_net.parameters():
            p.requires_grad = True
        for p in self.v_dc_net.k_net.parameters():
            p.requires_grad = True
        for p in self.v_dc_net.p_net.parameters():
            p.requires_grad = True
        self.c_net.train()

    def training_step(self, batch, batch_idx):
        Xt, X_train_indice, c_observed = batch
        x_full = torch.cat(Xt, dim=1)
        c_pred = self.c_net(x_full)
        data_w = 1.0 + 0.09 * c_observed  # same heuristic
        data_pointwise = data_w * (c_pred - c_observed) ** 2

        pde_residual = self.ad_dc_net.pde_residual(Xt)
        pde_pointwise = pde_residual ** 2
        loss_list = [data_pointwise, pde_pointwise]

        if self.incompressible:
            X, _t = Xt
            div_v = self.v_dc_net.incompressible_residual(X)
            div_pointwise = div_v ** 2
            loss_list.append(div_pointwise)

        if batch_idx == 0:
            self.log('joint_D', self.ad_dc_net.D.item())
        super().training_step(self.c_net, X_train_indice, loss_list, Xt, batch_idx)