import torch
import torch.nn as nn
from modules.rba_resample_trainer import Net_RBAResample
from modules.dc_net import AD_DC_Net
from utils.forward_sim import advect_diffuse_forward_simulation
from utils.visualize import draw_colorful_slice_image
from utils.mlflow_logging import (
    log_histogram_artifact,
    log_image_artifact,
    should_log_validation_artifacts,
    validation_epoch_step,
)
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
                 enable_rbar=True,validate_v_slices=[66, 68, 70, 71]):
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
        self.validate_v_slices = self._build_safe_slice_indices(
            validate_v_slices,
            total_size=self.ad_dc_net.char_domain.domain_shape[2],
        )

    @staticmethod
    def _build_safe_slice_indices(requested_slices, total_size):
        total_size = int(total_size)
        if total_size <= 0:
            return [0]

        safe = []
        for idx in requested_slices:
            idx_i = int(np.clip(idx, 0, total_size - 1))
            if idx_i not in safe:
                safe.append(idx_i)

        if safe:
            return safe

        fallback = np.linspace(0, total_size - 1, num=min(4, total_size), dtype=int)
        return list(dict.fromkeys(int(v) for v in fallback))

    # accept v of mm/min (physical space), so that time like t_index and t_jump is appropriate.
    def validate_forward_step(self, vx, vy, vz, t_index, t_jump):
        char_domain = self.ad_dc_net.char_domain
        c_scale = float(getattr(self.ad_dc_net.c_net, "C_star", 1.0))
        start_c = self.ad_dc_net.c_net.gt_data[:, :, :, t_index] * c_scale
        end_c = self.ad_dc_net.c_net.gt_data[:, :, :, t_index + t_jump] * c_scale

        t_duration = float(char_domain.t[t_index + t_jump] - char_domain.t[t_index])
        
        if char_domain.DTI_or_coef.ndim == 0:
            D = float(self.ad_dc_net.D.detach().cpu().item())
        else:
            D = char_domain.DTI_or_coef * self.ad_dc_net.D
            D = D.permute(2, 3, 4, 0, 1).reshape(*D.shape[2:], 3, 3).detach().cpu().numpy()
        num_steps = max(1, int(np.ceil(t_duration / 0.5)))
        frames = advect_diffuse_forward_simulation(
            start_c,
            vx,
            vy,
            vz,
            D,
            t_duration,
            num_steps=num_steps,
            voxel_dims=tuple(float(v) for v in char_domain.pixdim),
        )
        z_slice = char_domain.domain_shape[2] // 2
        
        slices = [
            frames[0][:, :, z_slice],
            frames[num_steps // 2][:, :, z_slice],
            frames[-1][:, :, z_slice],
            end_c[:, :, z_slice],
            np.abs((frames[-1] - end_c)[:, :, z_slice]),
        ]
        slices = np.vstack(slices)
        slices = np.clip(slices, 0, None)
        slices = slices / (np.max(slices) + 1e-8)
        return slices

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
            if not should_log_validation_artifacts(self):
                return val_loss

            log_step = validation_epoch_step(self)
            if self.ad_dc_net.char_domain.DTI_or_coef.ndim == 0:
                self.log('Diffusivity mm^2 per min', self.ad_dc_net.D)
            else:
                self.log('Scale factor for DTI', self.ad_dc_net.D)
            
            c_vis = self.ad_dc_net.c_net.draw_concentration_slices()
            log_image_artifact(
                logger=self.logger,
                image=c_vis,
                image_key='val_C_compare',
                step=log_step,
            )
            rgb_img, vx, vy, vz = self.ad_dc_net.v_dc_net.draw_velocity_volume()
            log_image_artifact(
                logger=self.logger,
                image=rgb_img,
                image_key='val_v_quiver',
                step=log_step,
            )
            
            v_mag = np.sqrt(vx**2 + vy**2 + vz**2).reshape(self.ad_dc_net.char_domain.domain_shape[:3])

            slice_images = []
            for z_slice_idx in self.validate_v_slices:
                v_slice = v_mag[z_slice_idx, :, :].T
                slice_img_whc = draw_colorful_slice_image(v_slice, 'jet')
                slice_images.append(slice_img_whc)
            combined_slices_img = np.concatenate(slice_images, axis=1)
            log_image_artifact(
                logger=self.logger,
                image=combined_slices_img,
                image_key='val_v_mag_slices',
                step=log_step,
            )
            
            log_image_artifact(
                logger=self.logger,
                image=self.validate_forward_step(
                    vx,
                    vy,
                    vz,
                    t_index=0,
                    t_jump=max(1, min(5, self.ad_dc_net.char_domain.domain_shape[3] - 1)),
                ),
                image_key='val_adv_diff_step',
                step=log_step,
            )
            k_vis = self.ad_dc_net.v_dc_net.k_net.draw_physical_slices()
            log_image_artifact(
                logger=self.logger,
                image=k_vis,
                image_key='val_k_slices',
                step=log_step,
            )

            p_vis = self.ad_dc_net.v_dc_net.p_net.draw_physical_slices()
            log_image_artifact(
                logger=self.logger,
                image=p_vis,
                image_key='val_p_slices',
                step=log_step,
            )

            flag_v_mag = v_mag.flatten()
            flag_v_mag = np.log(flag_v_mag[flag_v_mag > 1e-9])

            if flag_v_mag.size > 0:
                log_histogram_artifact(
                    logger=self.logger,
                    values=flag_v_mag,
                    hist_key='val_v_hist',
                    step=log_step,
                )
            
            flat_k = self.ad_dc_net.v_dc_net.k_net.get_physical_volume().flatten()
            flat_k = np.log(flat_k[flat_k > 1e-10])
            if flat_k.size > 0:
                if np.min(flat_k) <= 0:
                    flat_k = flat_k - np.min(flat_k) + 1e-3
                log_histogram_artifact(
                    logger=self.logger,
                    values=flat_k,
                    hist_key='val_k_hist',
                    step=log_step,
                )
            
            flat_p = self.ad_dc_net.v_dc_net.p_net.get_physical_volume(min_base=0).flatten()
            log_histogram_artifact(
                logger=self.logger,
                values=flat_p,
                hist_key='val_p_hist',
                step=log_step,
            )

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
        #     k_vis = self.ad_dc_net.v_dc_net.k_net.draw_physical_slices()
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
        # if batch_idx == 0:
        #     p_vis = self.ad_dc_net.v_dc_net.p_net.draw_pressure_slices()
        #     self.logger.experiment.add_image('train_p_slices', p_vis, self.current_epoch, dataformats='HWC')

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
                 div_weight=1e-6,
                 learning_rate=1e-3,
                 rba_learning_rate=0.1,
                 rba_memory=0.999,
                 enable_rbar=True,
                 enable_td_weight=True,
                 advpde_loss_name="ad_pde_p",
                 validate_v_slices=[43, 45, 49, 50]):
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
                         enable_rbar=enable_rbar,
                         validate_v_slices=validate_v_slices)
        # Freeze c & K
        for p in self.ad_dc_net.c_net.parameters():
            p.requires_grad = False
        for p in self.ad_dc_net.v_dc_net.k_net.parameters():
            p.requires_grad = False
        # Enable p_net params
        for p in self.ad_dc_net.v_dc_net.p_net.parameters():
            p.requires_grad = True
        self.ad_dc_net.c_net.eval()

        self.enable_td_weight = enable_td_weight

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
        
        # WARNING: use time independent scaling according to config
        super().training_step(self.ad_dc_net.c_net if self.enable_td_weight else None, 
                              X_train_indice, loss_list, Xt if self.enable_td_weight else None, batch_idx)

class DCPINN_ADPDE_P_K(DCPINN_ADPDE_P):
    # unfreeze k_net to jointly optimize p and k
    train_phase = "ad_pde_p_k"
    def __init__(self,
                 ad_dc_net: AD_DC_Net,
                 num_train_points,
                 incompressible=False,
                 div_weight=1e-5,
                 learning_rate=1e-3,
                 rba_learning_rate=0.1,
                 rba_memory=0.999,
                 enable_rbar=True,
                 enable_td_weight=True,
                 validate_v_slices=[66, 68, 70, 71]):
        super().__init__(ad_dc_net,
                         num_train_points,
                         incompressible=incompressible,
                         div_weight=div_weight,
                         learning_rate=learning_rate,
                         rba_learning_rate=rba_learning_rate,
                         rba_memory=rba_memory,
                         enable_rbar=enable_rbar,
                         enable_td_weight=enable_td_weight,
                         advpde_loss_name=DCPINN_ADPDE_P_K.train_phase,
                         validate_v_slices=validate_v_slices)
        # Freeze c_net, Unfreeze k_net and p_net
        for c in self.ad_dc_net.c_net.parameters():
            c.requires_grad = False
        for k in self.ad_dc_net.v_dc_net.k_net.parameters():
            k.requires_grad = True
        for p in self.ad_dc_net.v_dc_net.p_net.parameters():
            p.requires_grad = True


# WARNING: when joint optimize c_net, still need to using denoised one.
class DCPINN_Joint(DCPINN_Base):
    """Joint optimization of c_net, k_net, p_net with data + PDE + optional divergence losses.
    Batch: (Xt, X_train_indice, c_observed)
    Default weights: data=1.0, pde=10.0, div=1.0 (if incompressible)
    """

    train_phase = "joint_ad_pde+joint_data"

    def __init__(self,
                 ad_dc_net: AD_DC_Net,
                 num_train_points,
                 incompressible=False,
                 data_weight=1.0,
                 pde_weight=1.0,
                 div_weight=1e-5,
                 learning_rate=1e-6,
                 rba_learning_rate=0.1,
                 rba_memory=0.999,
                 enable_rbar=True,
                 enable_td_weight=True,
                 validate_v_slices=[43, 45, 49, 50]):
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
                         enable_rbar=enable_rbar,
                         validate_v_slices=validate_v_slices)
        # Unfreeze all
        for p in self.ad_dc_net.c_net.parameters():
            p.requires_grad = True
        for p in self.ad_dc_net.v_dc_net.k_net.parameters():
            p.requires_grad = True
        for p in self.ad_dc_net.v_dc_net.p_net.parameters():
            p.requires_grad = True
        self.ad_dc_net.c_net.train()
        self.enable_td_weight = enable_td_weight

    def training_step(self, batch, batch_idx):
        Xt, X_train_indice, _ = batch
        
        # WARNING: by default use denoised data loss to calculate joint loss as it performs better.
        data_pointwise, _ , _ = self.ad_dc_net.c_net.calculate_denoise_loss(batch)

        pde_residual = self.ad_dc_net.pde_residual(Xt)
        pde_pointwise = pde_residual ** 2
        loss_list = [data_pointwise, pde_pointwise]

        if self.incompressible:
            X, _t = Xt
            div_v = self.ad_dc_net.v_dc_net.incompressible_residual(X)
            div_pointwise = div_v ** 2
            loss_list.append(div_pointwise)

        super().training_step(self.ad_dc_net.c_net if self.enable_td_weight else None
                              , X_train_indice, loss_list, Xt if self.enable_td_weight else None, batch_idx)
