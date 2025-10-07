from modules.c_net import C_Net
from modules.data_module import CharacteristicDomain
import numpy as np
from modules.positional_encoding import PositionalEncoding_Geo
from torch import nn
import torch
from utils.visualize import fixed_quiver_image  # added import


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
        self.grid_tensor = char_domain.get_characteristic_geodomain().to(torch.float32)

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

    def forward(self, x):
        x_spatial = x[:, :3]
        # if incompressible, use vector potential to compute velocity
        if self.incompressible:
            psi = self.v_net_raw(x_spatial)  # steady-assumption
            psi_x, psi_y, psi_z = torch.split(psi, 1, dim=1)
            vx = (
                torch.autograd.grad(
                    psi_z,
                    x[:, 1],
                    grad_outputs=torch.ones_like(psi_z),
                    create_graph=True,
                )[0]
                - torch.autograd.grad(
                    psi_y,
                    x[:, 2],
                    grad_outputs=torch.ones_like(psi_y),
                    create_graph=True,
                )[0]
            )
            vy = (
                torch.autograd.grad(
                    psi_x,
                    x[:, 2],
                    grad_outputs=torch.ones_like(psi_x),
                    create_graph=True,
                )[0]
                - torch.autograd.grad(
                    psi_z,
                    x[:, 0],
                    grad_outputs=torch.ones_like(psi_z),
                    create_graph=True,
                )[0]
            )
            vz = (
                torch.autograd.grad(
                    psi_y,
                    x[:, 0],
                    grad_outputs=torch.ones_like(psi_y),
                    create_graph=True,
                )[0]
                - torch.autograd.grad(
                    psi_x,
                    x[:, 1],
                    grad_outputs=torch.ones_like(psi_x),
                    create_graph=True,
                )[0]
            )
            vx, vy, vz = vx.unsqueeze(1), vy.unsqueeze(1), vz.unsqueeze(1)
        else:
            v = self.v_net_raw(x_spatial)
            vx, vy, vz = v[:, 0:1], v[:, 1:2], v[:, 2:3]
        return vx, vy, vz

    def draw_velocity_volume(self):
        """
        mask: (nx, ny, nz) using that in char_domain
        pixdim: voxel spacing (3,) for physical quiver scaling
        """
        mask_np = self.char_domain.mask
        data_shape = self.char_domain.domain_shape[:3]
        nx, ny, nz = data_shape[0], data_shape[1], data_shape[2]
        with torch.no_grad():
            grid = self.grid_tensor.to(next(self.v_net_raw.parameters()).device)
            vx, vy, vz = self.forward(grid)
            vx = vx.cpu().numpy().reshape((nx, ny, nz)) * self.char_domain.V_star[0] * mask_np
            vy = vy.cpu().numpy().reshape((nx, ny, nz)) * self.char_domain.V_star[1] * mask_np
            vz = vz.cpu().numpy().reshape((nx, ny, nz)) * self.char_domain.V_star[2] * mask_np
            rgb_img = fixed_quiver_image(vx, vy, vz, self.char_domain.pixdim)
            return rgb_img, vx, vy, vz


# advection-diffusion network
class AD_Net(nn.Module):
    # accept dti (x,y,z,3,3) as input to compute anisotropic diffusion tensor
    # data,C_star is only for visualization inside c_net and v_net
    def __init__(
        self,
        c_layers,
        u_layers,
        data,C_star,
        char_domain : CharacteristicDomain,
        freq_nums=(8,8,8,0),
        incompressible=False,
        positional_encoding=True,
        gamma_space=1.0,
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
        # define learnable diffusivity
        self._log_Pe = nn.Parameter(torch.log(torch.tensor(char_domain.Pe_g)))

    @property
    def Pe(self):
        """Peclet number, representing the ratio of advection to diffusion."""
        return torch.exp(self._log_Pe)

    # used for learning
    @property
    def D_normalized(self):
        return 1.0 / self.Pe
    
    # used for visualization and result showing.
    @property
    def D(self):
        return self.D_normalized * (
            self.char_domain.L_star.mean() * 
            self.char_domain.V_star.mean())

    def forward(self, x):
        # x: (N,4) with (x,y,z,t) normalized to [0,1]
        c = self.c_net(x)
        vx, vy, vz = self.v_net(x)
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
        vx, vy, vz = self.v_net(torch.cat(Xt, dim=1))
        # When DTI is not known, apply anisotropic diffusion on grad_C for every timestep
        # provide learnable param char_domain.DTI_or_coef, cannot provide c as must be computed inside
        c_X, c_t, c_diffusion = self.c_net.get_c_grad_ani_diffusion(Xt, self.D_normalized)
        c_x, c_y, c_z = c_X[:, 0:1], c_X[:, 1:2], c_X[:, 2:3]
        # Advection-diffusion equation residual
        advection = vx * c_x + vy * c_y + vz * c_z
        return c_t + advection - c_diffusion
