import numpy as np
from torch import nn
import torch
from modules.c_net import MLP, C_Net
from modules.ad_net import V_Net
from modules.data_module import CharacteristicDomain

# in dc_net there is nn to build Darcy's law equation.
# We alias pressure field to C_Net (scalar spatio-temporal style, time ignored if domain time=1)
# and permeability to either C_Net (isotropic) or V_Net (anisotropic) for code reuse.

class P_Net(C_Net):
    """Pressure network: alias of C_Net (scalar field)."""
    def __init__(self, p_layers, data, char_domain: CharacteristicDomain, P_star,
                 positional_encoding=True, freq_nums=(8,8,8,0), gamma_space=1.0):
        super().__init__(p_layers, data, char_domain, P_star,
                         positional_encoding=positional_encoding,
                         freq_nums=freq_nums,
                         gamma_space=gamma_space)

class K_Net(nn.Module):
    """Permeability network:
    - Isotropic: alias C_Net producing positive scalar via exp
    - Anisotropic diagonal: alias V_Net (incompressible=False) producing 3 components, each exponentiated
    """
    def __init__(self, k_layers, data, char_domain: CharacteristicDomain, K_star,
                 positional_encoding=True, freq_nums=(8,8,8,0), gamma_space=1.0,
                 anisotropic=True):
        super().__init__()
        self.anisotropic = anisotropic
        self.char_domain = char_domain
        if anisotropic:
            # Use V_Net (3 outputs). Ensure not incompressible so raw outputs independent.
            self.raw_net = V_Net(k_layers, char_domain,
                                 positional_encoding=positional_encoding,
                                 freq_nums=freq_nums,
                                 incompressible=False,
                                 gamma_space=gamma_space)
        else:
            self.raw_net = C_Net(k_layers, data, char_domain, K_star,
                                 positional_encoding=positional_encoding,
                                 freq_nums=freq_nums,
                                 gamma_space=gamma_space)

    def forward(self, x):
        if self.anisotropic:
            kx, ky, kz = self.raw_net(x)
            return torch.exp(kx), torch.exp(ky), torch.exp(kz)  # enforce positivity
        else:
            k_iso = self.raw_net(x)
            return torch.exp(k_iso)  # positivity

class V_DC_Net(nn.Module):
    """Darcy's law network composing permeability (K_Net) and pressure (P_Net).
    Provides a velocity field via v = - grad p / k (component-wise for diagonal anisotropy).
    """
    def __init__(self,
                 p_layers,
                 k_layers,
                 data,  # reuse data for slice visualization
                 char_domain: CharacteristicDomain,
                 K_star, P_star,
                 positional_encoding=True,
                 freq_nums=(8,8,8,0),
                 gamma_space=1.0,
                 anisotropic=True):
        super().__init__()
        # Pressure and permeability networks
        self.p_net = P_Net(p_layers, data, char_domain, P_star,
                           positional_encoding=positional_encoding,
                           freq_nums=freq_nums,
                           gamma_space=gamma_space)
        self.k_net = K_Net(k_layers, data, char_domain, K_star,
                           positional_encoding=positional_encoding,
                           freq_nums=freq_nums,
                           gamma_space=gamma_space,
                           anisotropic=anisotropic)
        self.anisotropic = anisotropic

    def forward(self, x):
        # x: (N,3) spatial points (assumed normalized like other nets)
        x.requires_grad_(True)
        p = self.p_net(x)  # (N,1)

        # create graph for potential higher order derivatives
        grad_p = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), 
                                     create_graph=True)[0]
        if self.anisotropic:
            kx, ky, kz = self.k_net(x)  # each (N,1)
            v = -torch.cat([grad_p[:, 0:1] / kx,
                            grad_p[:, 1:2] / ky,
                            grad_p[:, 2:3] / kz], dim=1)
        else:
            k = self.k_net(x)  # (N,1)
            v = -grad_p / k
        return v

    # divergence of velocity for incompressibility check
    def incompressible_residual(self, x):
        v = self.forward(x)
        # not create graph as no higher order grad. retain graph for back propagate later. 
        v_grad = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), 
                                     retain_graph=True)[0]
        div_v = v_grad.sum(dim=1, keepdim=True)
        return div_v

# not only has V_DC_Net for V, but also has c_net to implement advect-diffuse residual 
class AD_DC_Net(nn.Module):
    """Advection-Diffusion network using Darcy-based velocity (V_DC_Net) and concentration (C_Net).
    Mirrors AD_Net but substitutes V_DC_Net for V_Net so that velocity obeys Darcy's law implicitly.

    PDE residual implemented for: dc/dt + v·grad(c) - D * (div(k grad c) / k?) ~ here we retain same form as AD_Net
    assuming diffusion tensor is provided via char_domain.DTI_or_coef (already scaled externally when anisotropic).
    """
    def __init__(self,
                 c_layers, p_layers, k_layers, data, C_star, K_star, P_star,
                 char_domain: CharacteristicDomain,
                 freq_nums=(8,8,8,0),
                 positional_encoding=True,
                 gamma_space=1.0,
                 anisotropic=True):
        super().__init__()
        freq_nums = np.array(freq_nums, dtype=int)
        self.char_domain = char_domain
        # Concentration network (scalar spatio-temporal field)
        self.c_net = C_Net(c_layers, data, char_domain, C_star,
                           positional_encoding=positional_encoding,
                           freq_nums=freq_nums,
                           gamma_space=gamma_space)
        # Darcy-based velocity network
        self.v_dc_net = V_DC_Net(p_layers,
                                 k_layers,
                                 data,
                                 char_domain,
                                 K_star,
                                 P_star,
                                 positional_encoding=positional_encoding,
                                 freq_nums=freq_nums,
                                 gamma_space=gamma_space,
                                 anisotropic=anisotropic)
        # Learnable Peclet number log parameter
        self._log_Pe = nn.Parameter(torch.log(torch.tensor(char_domain.Pe_g)))

    # Peclet number
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
        """x: (N,4) -> returns c, v (vx,vy,vz)"""
        c = self.c_net(x)
        v = self.v_dc_net(x[:, :3])  # velocity depends only on spatial part
        return c, v

    def pde_residual(self, Xt):
        """Compute advection-diffusion PDE residual at sample points.
        Xt: tuple (X, t) with X:(N,3), t:(N,1)
        Returns residual shape (N,1).
        """
        X, t = Xt
        v = self.v_dc_net(X)  # (N,3)
        # Scale diffusion tensor/coef by learnable D normalization (Pe)
        DTI_or_coef = self.D_normalized * self.char_domain.DTI_or_coef
        c_X, c_t, c_diffusion = self.c_net.get_c_grad_ani_diffusion((X, t), DTI_or_coef)
        c_x, c_y, c_z = c_X[:, 0:1], c_X[:, 1:2], c_X[:, 2:3]
        advection = v[:, 0:1] * c_x + v[:, 1:2] * c_y + v[:, 2:3] * c_z
        return c_t + advection - c_diffusion
