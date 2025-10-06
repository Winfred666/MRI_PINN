import numpy as np
from torch import nn
import torch
from modules.c_net import MLP, C_Net
from modules.ad_net import V_Net
from modules.data_module import CharacteristicDomain

# ---------------------- NEW / REWORKED P_Net (steady, spatial only) ----------------------
class P_Net(nn.Module):
    """
    Steady (time–independent) pressure network.
    - Accepts only spatial (x,y,z) input (shape (N,3) or (N,4) where last column (t) is ignored).
    - Uses geometric (spatial) positional encoding (no time harmonics).
    """
    def __init__(self,
                 p_layers,
                 char_domain: CharacteristicDomain,
                 P_star,
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
            # Each spatial axis gets: original + (sin,cos)*freq_count
            pe_extra = 2 * sum(self.freq_nums)
            in_dim = 3 + pe_extra

        self.mlp = MLP(p_layers, in_dim=in_dim, out_dim=1)
        
        self.val_slice_z = [char_domain.domain_shape[2] // 2 - 6, char_domain.domain_shape[2] // 2, char_domain.domain_shape[2] // 2 + 6]
        self.val_slice_t = [char_domain.domain_shape[3] // 4 * 2]
        # Clamp to valid range if domain has bounds
        if hasattr(char_domain, "t_index_max"):
            self.val_slice_t = [int(np.clip(t, 0, char_domain.t_index_max)) for t in self.val_slice_t]
        # Build 4D (X,Y,Z,T) sample points using existing helper; only Z,T vary (X,Y maybe mid-plane if helper does so)
        self.val_slice_4d = char_domain.get_characteristic_geotimedomain(
            slice_zindex=self.val_slice_z,
            slice_tindex=self.val_slice_t
        )

    def forward(self, x):
        # x can be (N,3) or (N,4); if (N,4) ignore time column
        if x.shape[-1] == 4:
            x = x[:, :3]
        # assume x already in characteristic (normalized) domain if upstream matches C_Net usage
        x_enc = self._positional_encode(x)
        p_hat = self.mlp(x_enc)
        return self.P_star * p_hat  # scale to physical magnitude (follow your earlier scaling pattern)

    def draw_pressure_slice(self, device=None):
        """
        Evaluate pressure over precomputed val_slice_4d (call prepare_pressure_slice first).
        Returns:
            P_vals: (num_points, 1) torch.Tensor (detached cpu)
        """
        if not hasattr(self, "val_slice_4d"):
            raise RuntimeError("Call prepare_pressure_slice() before draw_pressure_slice().")
        X4 = torch.as_tensor(self.val_slice_4d, dtype=torch.float32, device=device)
        with torch.no_grad():
            p = self.forward(X4[:, :3]).cpu()
        return p

# ---------------------- Modified K_Net ----------------------
class K_Net(nn.Module):
    """
    Permeability network:
    - Anisotropic diagonal: uses V_Net (3 components) -> each exponentiated for positivity.
    - Isotropic: uses steady spatial-only P_Net style (time independent) -> exponentiate scalar.
    """
    def __init__(self, k_layers, char_domain: CharacteristicDomain, K_star,
                 positional_encoding=True, freq_nums=(8,8,8,0), gamma_space=1.0,
                 anisotropic=True):
        super().__init__()
        self.anisotropic = anisotropic
        self.char_domain = char_domain
        if anisotropic:
            self.raw_net = V_Net(k_layers,
                                 char_domain,
                                 positional_encoding=positional_encoding,
                                 freq_nums=freq_nums,
                                 incompressible=False,
                                 gamma_space=gamma_space)
        else:
            # Reuse steady spatial-only pattern (P_Net-like) for isotropic permeability
            self.raw_net = P_Net(k_layers,
                                 char_domain,
                                 P_star=K_star,
                                 positional_encoding=positional_encoding,
                                 freq_nums=freq_nums,
                                 gamma_space=gamma_space)

    def forward(self, x):
        if self.anisotropic:
            kx, ky, kz = self.raw_net(x)  # expect each shape (N,1)
            return torch.exp(kx), torch.exp(ky), torch.exp(kz)
        else:
            if x.shape[-1] == 4:
                x = x[:, :3]
            k_iso = self.raw_net(x)
            return torch.exp(k_iso)

# ---------------------- Updated V_DC_Net (no data passed) ----------------------
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
                 anisotropic=True):
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

    def forward(self, x):
        # x: (N,3) or (N,4) -> use spatial part only
        if x.shape[-1] == 4:
            x = x[:, :3]
        x.requires_grad_(True)
        p = self.p_net(x)  # (N,1)
        grad_p = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p),
                                     create_graph=True)[0]  # (N,3)
        if self.anisotropic:
            kx, ky, kz = self.k_net(x)
            v = -torch.cat([grad_p[:, 0:1] / kx,
                            grad_p[:, 1:2] / ky,
                            grad_p[:, 2:3] / kz], dim=1)
        else:
            k = self.k_net(x)
            v = -grad_p / k
        return v

    def incompressible_residual(self, x):
        if x.shape[-1] == 4:
            x = x[:, :3]
        v = self.forward(x)
        v_grad = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                                     retain_graph=True)[0]
        div_v = v_grad.sum(dim=1, keepdim=True)
        return div_v

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
                 anisotropic=True):
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

        self._log_Pe = nn.Parameter(torch.log(torch.tensor(char_domain.Pe_g)))

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
        return self.D_normalized * (self.char_domain.L_star.mean() * self.char_domain.V_star.mean())

    def forward(self, x):
        """
        x: (N,4) (x,y,z,t)
        Returns (c, v) with:
          c: (N,1)
          v: (N,3) (steady, depends only on spatial subset)
        """
        c = self.c_net(x)
        v = self.v_dc_net(x[:, :3])
        return c, v

    def pde_residual(self, Xt):
        """Compute advection-diffusion PDE residual at sample points.
        Xt: tuple (X, t) with X:(N,3), t:(N,1)
        Returns residual shape (N,1).
        """
        X, t = Xt
        v = self.v_dc_net(X)  # (N,3)
        DTI_or_coef = self.D_normalized * self.char_domain.DTI_or_coef
        c_X, c_t, c_diffusion = self.c_net.get_c_grad_ani_diffusion((X, t), DTI_or_coef)
        c_x, c_y, c_z = c_X[:, 0:1], c_X[:, 1:2], c_X[:, 2:3]
        advection = v[:, 0:1] * c_x + v[:, 1:2] * c_y + v[:, 2:3] * c_z
        return c_t + advection - c_diffusion
