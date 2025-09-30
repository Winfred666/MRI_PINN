import numpy as np
from positional_encoding import PositionalEncoding_Geo, PositionalEncoding_GeoTime
from torch import nn
import torch


class C_Net(nn.Module):
    def __init__(
        self,
        c_layers,
        positional_encoding=True,
        domain_shape=(64, 64, 32, 10),
        gamma_space=1.0,
    ):
        super().__init__()
        self.c_layers = c_layers
        self.positional_encoding = positional_encoding

        if positional_encoding:
            num_freq_space = np.array(domain_shape[:3]) // 3
            num_freq_time = 0  # use fewer frequency for time dimension
            # always include input as non periodic representation.
            c_pos_encoder = PositionalEncoding_GeoTime(
                num_freq_space,
                num_freq_time,
                include_input=True,
                gamma_space=gamma_space,
            )
            # update input layer size
            c_layers[0] = 4 + num_freq_space.sum() * 2 + num_freq_time * 2

        self.c_net = nn.Sequential()
        if positional_encoding:
            self.c_net.add_module("c_positional_encoding", c_pos_encoder)
        self.c_net.add_module("c_input_layer", nn.Linear(c_layers[0], c_layers[1]))
        self.c_net.add_module("c_input_activation", nn.Tanh())
        for i in range(1, len(c_layers) - 2):
            self.c_net.add_module(
                f"c_hidden_layer_{i}", nn.Linear(c_layers[i], c_layers[i + 1])
            )
            self.c_net.add_module(f"c_hidden_activation_{i}", nn.Tanh())
        self.c_net.add_module("c_output_layer", nn.Linear(c_layers[-2], c_layers[-1]))

        for m in self.c_net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.c_net(x)


class V_Net(nn.Module):
    def __init__(
        self,
        v_layers,
        positional_encoding=True,
        domain_shape=(64, 64, 32, 10),
        incompressible=False,
        gamma_space=1.0,
    ):
        super().__init__()
        self.v_layers = v_layers
        self.positional_encoding = positional_encoding
        self.incompressible = incompressible

        if positional_encoding:
            num_freq_space = np.array(domain_shape[:3]) // 3
            # always include input as non periodic representation.
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
        else:
            v = self.v_net_raw(x_spatial)
            vx, vy, vz = v[:, 0:1], v[:, 1:2], v[:, 2:3]
        return vx, vy, vz


class AD_Net(nn.Module):
    def __init__(
        self,
        c_layers,
        u_layers,
        domain_shape,
        incompressible=False,
        positional_encoding=True,
        gamma_space=1.0,
    ):
        super().__init__()
        self.c_net = C_Net(
            c_layers, positional_encoding=positional_encoding, domain_shape=domain_shape, gamma_space=gamma_space
        )
        self.v_net = V_Net(
            u_layers,
            positional_encoding=positional_encoding,
            domain_shape=domain_shape,
            incompressible=incompressible,
            gamma_space=gamma_space,
        )

        # define learnable diffusivity
        self._D_raw = nn.Parameter(torch.log(torch.tensor(0.01)))

    @property
    def D(self):
        return torch.exp(self._D_raw)

    def forward(self, x):
        # x: (N,4) with (x,y,z,t) normalized to [0,1]
        c = self.c_net(x)
        vx, vy, vz = self.v_net(x)
        return c, vx, vy, vz

    def pde_residual(self, x):
        x = x.requires_grad_()
        C_pred, vx, vy, vz = self.forward(x)

        # Compute gradients of density
        grad_C = torch.autograd.grad(
            C_pred, x, grad_outputs=torch.ones_like(C_pred), create_graph=True
        )[0]
        C_x, C_y, C_z, C_t = (
            grad_C[:, 0:1],
            grad_C[:, 1:2],
            grad_C[:, 2:3],
            grad_C[:, 3:4],
        )

        # Second derivatives
        grad_C_x = torch.autograd.grad(
            C_x, x, grad_outputs=torch.ones_like(C_x), create_graph=True
        )[0]
        C_xx = grad_C_x[:, 0:1]

        grad_C_y = torch.autograd.grad(
            C_y, x, grad_outputs=torch.ones_like(C_y), create_graph=True
        )[0]
        C_yy = grad_C_y[:, 1:2]

        grad_C_z = torch.autograd.grad(
            C_z, x, grad_outputs=torch.ones_like(C_z), create_graph=True
        )[0]
        C_zz = grad_C_z[:, 2:3]

        # Advection-diffusion equation residual
        return C_t + vx * C_x + vy * C_y + vz * C_z - self.D * (C_xx + C_yy + C_zz)
