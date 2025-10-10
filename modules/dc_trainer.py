import torch
import torch.nn as nn
from modules.rba_resample_trainer import Net_RBAResample
from modules.dc_net import AD_DC_Net
from utils.forward_sim import advect_diffuse_forward_simulation
import numpy as np

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
        # FIX: Remove direct shortcut assignments to sub-modules.
        # self.c_net = ad_dc_net.c_net
        # self.v_dc_net = ad_dc_net.v_dc_net
        self.learning_rate = learning_rate
        self.L2_loss = nn.MSELoss()

    def validate_forward_step(self, vx, vy, vz, t_index, t_jump):
        # transfer v back to mm/min, so that time like t_index and t_jump is appropriate.
        char_domain = self.ad_dc_net.char_domain
        vx, vy, vz = vx * char_domain.V_star[0], vy * char_domain.V_star[1], vz * char_domain.V_star[2]
        # visualize how concentration field advects and diffuses in one step
        # get spatial points at t_index and t_index + 1
        start_c = self.ad_dc_net.c_net.gt_data[:, :, :, t_index]
        end_c = self.ad_dc_net.c_net.gt_data[:, :, :, t_index + t_jump]

        # we are not using training c_dataset (half of whole data), so no need to *2 for duration.
        t_duration = char_domain.t[t_index + t_jump] - char_domain.t[t_index]  # in min
        
        # D is the real diffusion coefficient(tensor with ndim 0 or (nx,ny,nz,3,3)), in mm^2/min
        if char_domain.DTI_or_coef.ndim == 0:
            D = self.ad_dc_net.D.detach().cpu().numpy()
        else:
            D = char_domain.DTI_or_coef * self.ad_dc_net.D
            # reshape (1, 9, Nx, Ny, Nz) back to (nx,ny,nz,3,3)
            D = D.permute(2, 3, 4, 0, 1).reshape(*D.shape[2:], 3, 3).detach().cpu().numpy()
        num_steps = int(t_duration * 10)  # 0.1 min per step
        frames = advect_diffuse_forward_simulation(start_c, vx, vy, vz, D, t_duration,
                                                   num_steps=num_steps,
                                                   voxel_dims=char_domain.pixdim) # 4 min between frame.
        z_slice = char_domain.domain_shape[2] // 2
        
        # now return a stack of volume slice, start, end and error, ready for tensorboard
        slices = [frames[0][:,:,z_slice], frames[num_steps//2][:,:,z_slice], 
                  frames[-1][:,:,z_slice], end_c[:,:,z_slice], (frames[-1]-end_c)[:,:,z_slice]]
        return np.vstack(slices)  # shape (4, nx, ny)

    def validation_step(self, batch, batch_idx):
        # always use DCE-MRI batch for validation
        Xt, _, c_observed = batch
        x_full = torch.cat(Xt, dim=1)
        with torch.no_grad():
            # FIX: Access c_net through the main ad_dc_net module.
            c_pred = self.ad_dc_net.c_net(x_full)
        val_loss = ((c_pred - c_observed) ** 2).mean()
        if batch_idx == 0:
            self.log('val_data_loss', val_loss)
            if self.ad_dc_net.char_domain.DTI_or_coef.ndim == 0:
                self.log('Diffusivity mm^2 per min', self.ad_dc_net.D)
            else:
                self.log('Scale factor for DTI', self.ad_dc_net.D)
            # FIX: Access sub-modules through the main ad_dc_net module.
            c_vis = self.ad_dc_net.c_net.draw_concentration_slices()
            self.logger.experiment.add_image('val_C_compare', c_vis, self.current_epoch, dataformats='WH')
            rgb_img, vx, vy, vz = self.ad_dc_net.v_dc_net.draw_velocity_volume()
            self.logger.experiment.add_image('val_v_quiver', rgb_img, self.current_epoch, dataformats='HWC')
            
            self.logger.experiment.add_image('val_adv_diff_step', 
                                             self.validate_forward_step(vx, vy, vz, t_index=0, 
                                                                        t_jump=5), self.current_epoch, dataformats='WH')
            k_vis = self.ad_dc_net.v_dc_net.k_net.draw_permeability_volume() # shaped (H,W,C)
            self.logger.experiment.add_image('val_K_slices', k_vis, self.current_epoch, dataformats='HWC')
            p_vis = self.ad_dc_net.v_dc_net.p_net.draw_pressure_slices()
            self.logger.experiment.add_image('val_p_slices', p_vis, self.current_epoch, dataformats='HWC')


        return val_loss


class DCPINN_InitK(DCPINN_Base):
    train_phase = "init_k_data"

    def __init__(self,
                 ad_dc_net: AD_DC_Net,
                 num_train_points,
                 learning_rate=1e-3,
                 rba_learning_rate=0.1,
                 rba_memory=0.999,
                 enable_rbar=True):
        super().__init__(ad_dc_net,
                         num_train_points,
                         rba_name_weight_list=[(DCPINN_InitK.train_phase, 1.0)],
                         learning_rate=learning_rate,
                         rba_learning_rate=rba_learning_rate,
                         rba_memory=rba_memory,
                         enable_rbar=enable_rbar)
        # self.k_net = ad_dc_net.v_dc_net.k_net # This was already correctly identified as a problem.

    def training_step(self, batch, batch_idx):
        X, X_train_indice, k_observed = batch
        # FIX: Access k_net through the main ad_dc_net module.
        k_pred = self.ad_dc_net.v_dc_net.k_net(X)
        pointwise_loss = ((k_pred - k_observed) ** 2).mean(dim=1, keepdim=True)
        super().training_step(None, X_train_indice, [pointwise_loss], None, batch_idx)

        # DEBUG: visualize k field slice to see if it is learning
        # if batch_idx == 0:
        #     # FIX: Access k_net through the main ad_dc_net module.
        #     k_vis = self.ad_dc_net.v_dc_net.k_net.draw_permeability_volume()
        #     self.logger.experiment.add_image('train_K_slices', k_vis, self.current_epoch, dataformats='HWC')

# use velocity datamodule to init p net
class DCPINN_InitP(DCPINN_Base):
    train_phase = "init_p_data"
    def __init__(self,
                 ad_dc_net: AD_DC_Net,
                 num_train_points,
                 learning_rate=1e-3,
                 rba_learning_rate=0.1,
                 rba_memory=0.999,
                 enable_rbar=True):
        super().__init__(ad_dc_net,
                         num_train_points,
                         rba_name_weight_list=[(DCPINN_InitP.train_phase, 1.0)],
                         learning_rate=learning_rate,
                         rba_learning_rate=rba_learning_rate,
                         rba_memory=rba_memory,
                         enable_rbar=enable_rbar)
        # self.v_dc_net = ad_dc_net.v_dc_net # FIX: Remove this shortcut.
        # Freeze K net, Unfreeze P net
        for p in self.ad_dc_net.v_dc_net.k_net.parameters():
            p.requires_grad = False
        for p in self.ad_dc_net.v_dc_net.p_net.parameters():
            p.requires_grad = True
    
    def training_step(self, batch, batch_idx):
        X, X_train_indice, v_observed = batch
        # FIX: Access v_dc_net through the main ad_dc_net module.
        v_pred_list = self.ad_dc_net.v_dc_net(X)
        v_pred = torch.cat(v_pred_list, dim=1)  # (N,3)
        pointwise_loss = ((v_pred - v_observed) ** 2).mean(dim=1, keepdim=True)
        
        super().training_step(None, X_train_indice, [pointwise_loss], None, batch_idx)

        # DEBUG: visualize p field slice to see if it is learning
        if batch_idx == 0:
            p_vis = self.ad_dc_net.v_dc_net.p_net.draw_pressure_slices()
            self.logger.experiment.add_image('train_p_slices', p_vis, self.current_epoch, dataformats='HWC')

class DCPINN_ADPDE_P(DCPINN_Base):
    """Optimize pressure network via advection-diffusion residual (+ optional divergence residual).
    Batch: (Xt, X_train_indice, c_dummy) – c_dummy unused.
    RBA entries: [('pde_p_residual', 1.0)] or plus ('div_v_residual', div_weight) if incompressible.
    """
    train_phase = "ad_pde_p"

    def __init__(self,
                 ad_dc_net: AD_DC_Net,
                 num_train_points,
                 incompressible=False,
                 div_weight=1.0,
                 learning_rate=1e-3,
                 rba_learning_rate=0.1,
                 rba_memory=0.999,
                 enable_rbar=True,
                 advpde_loss_name="ad_pde_p"):
        rba_list = [(advpde_loss_name, 1.0)]
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
        for p in self.ad_dc_net.c_net.parameters():
            p.requires_grad = False
        for p in self.ad_dc_net.v_dc_net.k_net.parameters():
            p.requires_grad = False
        # Enable p_net params
        for p in self.ad_dc_net.v_dc_net.p_net.parameters():
            p.requires_grad = True
        self.ad_dc_net.c_net.eval()

    def training_step(self, batch, batch_idx):
        Xt, X_train_indice, _ = batch
        pde_residual = self.ad_dc_net.pde_residual(Xt)  # (N,1)
        pde_pointwise = pde_residual ** 2
        loss_list = [pde_pointwise]
        if self.incompressible:
            X, _ = Xt
            div_v = self.ad_dc_net.v_dc_net.incompressible_residual(X)  # (N,1)
            div_pointwise = div_v ** 2
            loss_list.append(div_pointwise)
        
        super().training_step(self.ad_dc_net.c_net, X_train_indice, loss_list, Xt, batch_idx)

class DCPINN_ADPDE_P_K(DCPINN_ADPDE_P):
    # unfreeze k_net to jointly optimize p and k
    train_phase = "ad_pde_p_k"
    def __init__(self,
                 ad_dc_net: AD_DC_Net,
                 num_train_points,
                 incompressible=False,
                 div_weight=1.0,
                 learning_rate=1e-3,
                 rba_learning_rate=0.1,
                 rba_memory=0.999,
                 enable_rbar=True):
        super().__init__(ad_dc_net,
                         num_train_points,
                         incompressible=incompressible,
                         div_weight=div_weight,
                         learning_rate=learning_rate,
                         rba_learning_rate=rba_learning_rate,
                         rba_memory=rba_memory,
                         enable_rbar=enable_rbar,
                         advpde_loss_name=DCPINN_ADPDE_P_K.train_phase)
        # Freeze c_net, Unfreeze k_net and p_net
        for c in self.ad_dc_net.c_net.parameters():
            c.requires_grad = False
        for k in self.ad_dc_net.v_dc_net.k_net.parameters():
            k.requires_grad = True
        for p in self.ad_dc_net.v_dc_net.p_net.parameters():
            p.requires_grad = True


class DCPINN_Joint(DCPINN_Base):
    """Joint optimization of c_net, k_net, p_net with data + PDE + optional divergence losses.
    Batch: (Xt, X_train_indice, c_observed)
    Default weights: data=1.0, pde=10.0, div=1.0 (if incompressible)
    """

    train_phase = "joint_data+joint_ad_pde"

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
        rba_list = [("joint_data", data_weight), ("joint_ad_pde", pde_weight)]
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
        for p in self.ad_dc_net.c_net.parameters():
            p.requires_grad = True
        for p in self.ad_dc_net.v_dc_net.k_net.parameters():
            p.requires_grad = True
        for p in self.ad_dc_net.v_dc_net.p_net.parameters():
            p.requires_grad = True
        self.ad_dc_net.c_net.train()

    def training_step(self, batch, batch_idx):
        Xt, X_train_indice, c_observed = batch
        x_full = torch.cat(Xt, dim=1)
        c_pred = self.ad_dc_net.c_net(x_full)
        data_w = 1.0 + 0.09 * c_observed  # same heuristic
        data_pointwise = data_w * (c_pred - c_observed) ** 2

        pde_residual = self.ad_dc_net.pde_residual(Xt)
        pde_pointwise = pde_residual ** 2
        loss_list = [data_pointwise, pde_pointwise]

        if self.incompressible:
            X, _t = Xt
            div_v = self.ad_dc_net.v_dc_net.incompressible_residual(X)
            div_pointwise = div_v ** 2
            loss_list.append(div_pointwise)

        super().training_step(self.ad_dc_net.c_net, X_train_indice, loss_list, Xt, batch_idx)