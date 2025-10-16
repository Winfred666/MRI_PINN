from modules.c_net import C_Net
from modules.data_module import CharacteristicDomain
import numpy as np
from modules.positional_encoding import PositionalEncoding_Geo
from torch import nn
import torch
from utils.visualize import fixed_quiver_image


# in ad_net there is nn to build advect-diffuse equation,
# we train the equation as pde loss 
# including a concentration network c_net and a velocity network v_net.

class V_Net(nn.Module):
    def __init__(
        self,
        v_layers,
        char_domain: CharacteristicDomain,
        positional_encoding=True,
        freq_nums=(8,8,8,0),
        incompressible=False,
        gamma_space=1.0,
    ):
        super().__init__()
        self.v_layers = v_layers
        self.positional_encoding = positional_encoding
        self.incompressible = incompressible
        self.char_domain = char_domain
        # precompute spatial grid (Nxyz,3)
        self.val_volume_3d = char_domain.get_characteristic_geodomain()

        if positional_encoding:
            num_freq_space = np.array(freq_nums[:3])
            v_pos_encoder = PositionalEncoding_Geo(
                num_freq_space, include_input=True, gamma_space=gamma_space
            )
            # update input layer size
            v_layers[0] = 3 + num_freq_space.sum() * 2

        self.v_net_raw = nn.Sequential()
        if positional_encoding:
            self.v_net_raw.add_module("v_positional_encoding", v_pos_encoder)
        self.v_net_raw.add_module("v_input_layer", nn.Linear(v_layers[0], v_layers[1]))
        self.v_net_raw.add_module("v_input_activation", nn.Tanh())
        for i in range(1, len(v_layers) - 2):
            self.v_net_raw.add_module(
                f"v_hidden_layer_{i}", nn.Linear(v_layers[i], v_layers[i + 1])
            )
            self.v_net_raw.add_module(f"v_hidden_activation_{i}", nn.Tanh())
        # The final layer outputs vx, vy, vz
        self.v_net_raw.add_module(
            "v_output_layer", nn.Linear(v_layers[-2], v_layers[-1])
        )

        for m in self.v_net_raw.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, X):
        # if incompressible, use vector potential to compute velocity
        if self.incompressible:
            X.requires_grad_(True)
            psi = self.v_net_raw(X)  # steady-assumption
            psi_x, psi_y, psi_z = torch.split(psi, 1, dim=1)
            psi_x_grad = torch.autograd.grad(
                psi_x, X,
                grad_outputs=torch.ones_like(psi_x),
                create_graph=True,
            )[0]
            psi_y_grad = torch.autograd.grad(
                psi_y, X,
                grad_outputs=torch.ones_like(psi_y),
                create_graph=True,
            )[0]
            psi_z_grad = torch.autograd.grad(
                psi_z, X,
                grad_outputs=torch.ones_like(psi_z),
                create_graph=True,
            )[0]
            vx = (
                psi_z_grad[:, 1]
                - psi_y_grad[:, 2]
            )
            vy = (
                psi_x_grad[:, 2]
                - psi_z_grad[:, 0]
            )
            vz = (
                psi_y_grad[:, 0]
                - psi_x_grad[:, 1]
            )
            vx, vy, vz = vx.unsqueeze(1), vy.unsqueeze(1), vz.unsqueeze(1)
        else:
            v = self.v_net_raw(X)
            vx, vy, vz = v[:, 0:1], v[:, 1:2], v[:, 2:3]
        return vx, vy, vz

    def draw_velocity_volume(self, label="|v| magnitude"):
        """
        mask: (nx, ny, nz) using that in char_domain
        pixdim: voxel spacing (3,) for physical quiver(unit mm/min) scaling
        """
        mask_np = self.char_domain.mask
        data_shape = self.char_domain.domain_shape[:3]
        nx, ny, nz = data_shape[0], data_shape[1], data_shape[2]
        with torch.no_grad():
            grid = self.val_volume_3d.to(next(self.v_net_raw.parameters()).device)
            vx, vy, vz = self.forward(grid)
            vx = vx.cpu().numpy().reshape((nx, ny, nz)) * self.char_domain.V_star[0] * mask_np
            vy = vy.cpu().numpy().reshape((nx, ny, nz)) * self.char_domain.V_star[1] * mask_np
            vz = vz.cpu().numpy().reshape((nx, ny, nz)) * self.char_domain.V_star[2] * mask_np
            rgb_img = fixed_quiver_image(vx, vy, vz, self.char_domain.pixdim, label=label)
            return rgb_img, vx, vy, vz


# advection-diffusion network
class AD_Net(nn.Module):
    # accept dti (x,y,z,3,3) as input to compute anisotropic diffusion tensor
    # data,C_star is only for visualization inside c_net and v_net
    def __init__(
        self,
        c_layers,
        u_layers,
        data, C_star,
        char_domain: CharacteristicDomain,
        freq_nums=(8,8,8,0),
        incompressible=False,
        positional_encoding=True,
        gamma_space=1.0,
        use_learnable_D=False,  # if False, keep D constant by storing _log_Pe as a buffer
    ):
        super().__init__()

        freq_nums = np.array(freq_nums, dtype=int)
        self.char_domain = char_domain  # store for D computation

        self.c_net = C_Net(
            c_layers, data, char_domain, C_star,
            positional_encoding=positional_encoding,
            freq_nums=freq_nums, gamma_space=gamma_space
        )

        self.v_net = V_Net(
            u_layers,
            char_domain,
            positional_encoding=positional_encoding,
            freq_nums=freq_nums,
            incompressible=incompressible,
            gamma_space=gamma_space,
        )

        # when use_learnable_D is False, keep it constant via register_buffer
        log_Pe_init = torch.log(torch.as_tensor(char_domain.Pe_g, dtype=torch.float32))
        if use_learnable_D:
            self.register_buffer("_log_Pe", log_Pe_init)
        else:
            self._log_Pe = nn.Parameter(log_Pe_init.clone())
        
    @property
    def Pe(self):
        """Peclet number, representing the ratio of advection to diffusion."""
        return torch.exp(self._log_Pe)

    # used for learning, characteristic domain
    @property
    def D_normalized(self):
        return 1.0 / self.Pe
    
    # used for visualization and result showing, unit mm^2/min
    @property
    def D(self):
        return self.D_normalized * (
            self.char_domain.L_star.mean() * 
            self.char_domain.V_star.mean())

    def forward(self, Xt):
        # x: (N,4) with (x,y,z,t) normalized to [0,1]
        c = self.c_net(Xt)
        vx, vy, vz = self.v_net(Xt[0])
        return c, vx, vy, vz

    def c_t_smoothness_residual(self, Xt, provided_pred_c=None):
        _, t = Xt
        t.requires_grad_(True)
        if provided_pred_c is None:
            C_pred = self.c_net(torch.cat(Xt, dim=1))
        else:
            C_pred = provided_pred_c
        c_t = torch.autograd.grad(
            C_pred, t, grad_outputs=torch.ones_like(C_pred), create_graph=True
        )[0]
        return (c_t **2).mean()

    def pde_residual(self, Xt):
        vx, vy, vz = self.v_net(Xt[0])
        # When DTI is not known, apply anisotropic diffusion on grad_C for every timestep
        # provide learnable param char_domain.DTI_or_coef, cannot provide c as must be computed inside
        c_X, c_t, c_diffusion = self.c_net.get_c_grad_ani_diffusion(Xt, self.D_normalized)
        c_x, c_y, c_z = c_X[:, 0:1], c_X[:, 1:2], c_X[:, 2:3]
        # Advection-diffusion equation residual
        advection = vx * c_x + vy * c_y + vz * c_z
        return c_t + advection - c_diffusion
