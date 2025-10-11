import numpy as np
from torch import nn
import torch
from modules.c_net import MLP, C_Net
from modules.ad_net import V_Net
from modules.data_module import CharacteristicDomain
from modules.positional_encoding import PositionalEncoding_Geo
from utils.visualize import fixed_quiver_image, draw_colorful_slice_image

class P_Net(nn.Module):
    """
    Steady (time–independent) pressure network.
    - Accepts only spatial (x,y,z) input (shape (N,3) or (N,4) where last column (t) is ignored).
    - Uses geometric (spatial) positional encoding (no time harmonics).
    """
    def __init__(self,
                 p_layers,
                 char_domain: CharacteristicDomain,
                 P_star, # (3,) array-like for x,y,z P_grad.
                 positional_encoding=True,
                 freq_nums=(8, 8, 8, 0),
                 gamma_space=1.0):
        super().__init__()
        self.char_domain = char_domain
        self.positional_encoding = positional_encoding
        self.freq_nums = tuple(int(f) for f in freq_nums[:3])  # only spatial counts
        self.gamma_space = gamma_space
        self.P_star = torch.as_tensor(P_star, dtype=torch.float32)

        in_dim = 3
        if positional_encoding:
            num_freq_space = np.array(freq_nums[:3])
            self.p_pos_encoder = PositionalEncoding_Geo(
                num_freq_space, include_input=True, gamma_space=gamma_space
            )
            # Each spatial axis gets: original + (sin,cos)*freq_count
            in_dim = 3 + 2 * sum(self.freq_nums)

        self.mlp = MLP(
            input_dim=in_dim,
            output_dim=p_layers[-1],
            hidden_layers=len(p_layers) - 2,
            hidden_features=p_layers[1],
        )
        
        self.val_slice_z = [char_domain.domain_shape[2] // 2 - 6, char_domain.domain_shape[2] // 2, char_domain.domain_shape[2] // 2 + 6]
        # Build 3D (X,Y,Z) sample points using existing helper; only Z vary (X,Y maybe mid-plane if helper does so)
        self.val_slice_3d = char_domain.get_characteristic_geodomain(slice_zindex=self.val_slice_z)
    

    # for steady state, uniformly pass X (N,3) of Xt.
    def forward(self, X):
        if self.positional_encoding:
            X = self.p_pos_encoder(X)
        return self.mlp(X)

    def draw_pressure_slices(self, cmap="inferno"):
        """
        Evaluate pressure over precomputed val_slice_3d (call prepare_pressure_slice first).
        Returns:
            P_vals: (num_points, 1) torch.Tensor (detached cpu)
        """
        with torch.no_grad():
            device = next(self.parameters()).device
            val_coords = self.val_slice_3d.to(device) # shape (nx*ny*num_slices, 3)
            vol_disp_all = self.forward(val_coords).cpu().numpy().reshape(
                self.char_domain.domain_shape[0], self.char_domain.domain_shape[1], len(self.val_slice_z)
            )
            p_vis_list = []
            for i in range(len(self.val_slice_z)):
                mask_disp = self.char_domain.mask[:, :, self.val_slice_z[i]]
                vol_disp = vol_disp_all[:, :, i] * mask_disp  # apply mask
                # for each slice, apply cmap using matplotlib and save as RGB
                image = draw_colorful_slice_image(vol_disp, cmap=cmap, mask=mask_disp) # shaped (nx, ny, 3)
                p_vis_list.append(image)
            return np.concatenate(p_vis_list, axis=1)  # concatenate along width (nx, ny*num_slices, 3)


class K_Net(nn.Module):
    """
    Permeability network:
    - Anisotropic diagonal: uses V_Net (3 components) -> each exponentiated for positivity.
    - Isotropic: uses steady spatial-only P_Net style (time independent) -> exponentiate scalar.
    """
    def __init__(self, k_layers, char_domain: CharacteristicDomain, K_star,
                 positional_encoding=True, freq_nums=(8,8,8,0), gamma_space=1.0,
                 anisotropic=False):
        super().__init__()
        self.anisotropic = anisotropic
        self.char_domain = char_domain
        if anisotropic:
            self.k_net = V_Net(k_layers,
                                 char_domain,
                                 positional_encoding=positional_encoding,
                                 freq_nums=freq_nums,
                                 incompressible=False,
                                 gamma_space=gamma_space)
        else:
            # Reuse steady spatial-only pattern (P_Net-like) for isotropic permeability
            self.k_net = P_Net(k_layers,
                                 char_domain,
                                 P_star=K_star,
                                 positional_encoding=positional_encoding,
                                 freq_nums=freq_nums,
                                 gamma_space=gamma_space)

    def forward(self, X):
        # already in characteristic domain, no need to scale,
        # also P_Net or V_Net would handle (N,3) or (N,4) input shapes
        return self.k_net(X)
    
    def draw_permeability_volume(self):
        if self.anisotropic:
            # use V_Net style drawing
            return self.k_net.draw_velocity_volume(label="|k| magnitude")[0]  # rgb_img
        else:
            # use P_Net style drawing
            return self.k_net.draw_pressure_slices(cmap="cool")  # P_vals

class V_DC_Net(nn.Module):
    """
    Darcy velocity network:
    v = - grad(p) / k  (component-wise for anisotropic diagonal permeability).
    Pressure & (optionally anisotropic) permeability are steady (spatial only).
    """
    def __init__(self,
                 p_layers,
                 k_layers,
                 char_domain: CharacteristicDomain,
                 K_star, P_star,
                 positional_encoding=True,
                 freq_nums=(8,8,8,0),
                 gamma_space=1.0,
                 anisotropic=False):
        super().__init__()
        self.p_net = P_Net(p_layers,
                           char_domain,
                           P_star=P_star,
                           positional_encoding=positional_encoding,
                           freq_nums=freq_nums,
                           gamma_space=gamma_space)
        self.k_net = K_Net(k_layers,
                           char_domain,
                           K_star,
                           positional_encoding=positional_encoding,
                           freq_nums=freq_nums,
                           gamma_space=gamma_space,
                           anisotropic=anisotropic)
        self.anisotropic = anisotropic
        self.char_domain = char_domain
        self.grid_tensor = char_domain.get_characteristic_geodomain().to(torch.float32)

    def forward(self, X):
        # this forward pass is called within a torch.no_grad() block.
        with torch.enable_grad():
            # X: (N,3) -> use spatial part only
            X.requires_grad_(True)
            # Temporarily enable gradient tracking to compute grad_p, even if
            p = self.p_net(X)  # (N,1)
            grad_p = torch.autograd.grad(p, X, grad_outputs=torch.ones_like(p),
                                         create_graph=True)[0]  # (N,3)
        if self.anisotropic:
            kx, ky, kz = self.k_net(X)
            vx = -grad_p[:, 0:1] * kx
            vy = -grad_p[:, 1:2] * ky
            vz = -grad_p[:, 2:3] * kz
        else:
            k = self.k_net(X)
            v = -grad_p * k
            vx, vy, vz = v[:, 0:1], v[:, 1:2], v[:, 2:3]

        return vx, vy, vz

    def incompressible_residual(self, X):
        if X.shape[-1] == 4:
            X = X[:, :3]
        vx, vy, vz = self.forward(X)
        # need retain_graph True to allow multiple calls within same graph (e.g. in pde_residual)
        v_grad_x = torch.autograd.grad(vx, X, grad_outputs=torch.ones_like(vx),
                                        retain_graph=True)[0]
        v_grad_y = torch.autograd.grad(vy, X, grad_outputs=torch.ones_like(vy),
                                        retain_graph=True)[0]
        v_grad_z = torch.autograd.grad(vz, X, grad_outputs=torch.ones_like(vz),
                                        retain_graph=True)[0]
        div_v = v_grad_x[:, 0:1] + v_grad_y[:, 1:2] + v_grad_z[:, 2:3]
        return div_v
    
    def draw_velocity_volume(self):
        """
        mask: (nx, ny, nz) using that in char_domain
        pixdim: voxel spacing (3,) for physical quiver scaling
        """
        mask_np = self.char_domain.mask
        data_shape = self.char_domain.domain_shape[:3]
        nx, ny, nz = data_shape[0], data_shape[1], data_shape[2]
        with torch.no_grad():
            grid = self.grid_tensor.to(next(self.p_net.parameters()).device)
            vx, vy, vz = self.forward(grid)
            vx = vx.cpu().numpy().reshape((nx, ny, nz)) * self.char_domain.V_star[0] * mask_np
            vy = vy.cpu().numpy().reshape((nx, ny, nz)) * self.char_domain.V_star[1] * mask_np
            vz = vz.cpu().numpy().reshape((nx, ny, nz)) * self.char_domain.V_star[2] * mask_np
            rgb_img = fixed_quiver_image(vx, vy, vz, self.char_domain.pixdim)
            return rgb_img, vx, vy, vz


# ---------------------- AD_DC_Net updated to use new V_DC_Net signature ----------------------
class AD_DC_Net(nn.Module):
    """
    Advection-Diffusion with Darcy velocity (steady pressure/permeability).
    c = c(x,y,z,t), v = v(x,y,z)
    """
    def __init__(self,
                 c_layers, p_layers, k_layers,
                 data, C_star, K_star, P_star,
                 char_domain: CharacteristicDomain,
                 freq_nums=(8,8,8,0),
                 positional_encoding=True,
                 gamma_space=1.0,
                 anisotropic=False,
                 use_learnable_D=False):
        super().__init__()
        freq_nums = np.array(freq_nums, dtype=int)
        self.char_domain = char_domain

        # Concentration network still spatio-temporal (original C_Net)
        self.c_net = C_Net(c_layers, data, char_domain, C_star,
                           positional_encoding=positional_encoding,
                           freq_nums=freq_nums,
                           gamma_space=gamma_space)

        # Steady Darcy velocity
        self.v_dc_net = V_DC_Net(p_layers,
                                 k_layers,
                                 char_domain,
                                 K_star,
                                 P_star,
                                 positional_encoding=positional_encoding,
                                 freq_nums=freq_nums,
                                 gamma_space=gamma_space,
                                 anisotropic=anisotropic)
    
        log_Pe_init = torch.log(torch.as_tensor(char_domain.Pe_g, dtype=torch.float32))
        if use_learnable_D:
            self.register_buffer("_log_Pe", log_Pe_init)
        else:
            self._log_Pe = nn.Parameter(log_Pe_init.clone())


    @property
    def Pe(self):
        return torch.exp(self._log_Pe)

    # Normalized diffusion coefficient (dimensionless)
    @property
    def D_normalized(self):
        return 1.0 / self.Pe

    # Physical diffusion (for logging / visualization) using same scaling heuristic as AD_Net
    @property
    def D(self):
        # if anisotropic, there is no D_coeff, only D_tensor, and D should be D_normalize which is scaling factor.
        if self.char_domain.DTI_or_coef.ndim == 0:
            return self.D_normalized * (self.char_domain.L_star.mean() * self.char_domain.V_star.mean())
        else:
            return self.D_normalized # just scaling factor, no physical meaning


    def forward(self, Xt):
        """
        x: (N,4) (x,y,z,t)
        Returns (c, v) with:
          c: (N,1)
          v: (N,3) (steady, depends only on spatial subset)
        """
        X, t = Xt
        c = self.c_net(torch.cat([X, t], dim=1))  # (N,1)
        vx,vy,vz = self.v_dc_net(X)  # (N,3)
        return c, vx,vy,vz

    def pde_residual(self, Xt):
        """Compute advection-diffusion PDE residual at sample points.
        Xt: tuple (X, t) with X:(N,3), t:(N,1)
        Returns residual shape (N,1).
        """
        X, t = Xt
        vx, vy, vz = self.v_dc_net(X)  # (N,3)
        c_X, c_t, c_diffusion = self.c_net.get_c_grad_ani_diffusion((X, t), self.D_normalized)
        c_x, c_y, c_z = c_X[:, 0:1], c_X[:, 1:2], c_X[:, 2:3]
        advection = vx * c_x + vy * c_y + vz * c_z
        return c_t + advection - c_diffusion
