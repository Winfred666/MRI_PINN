import numpy as np
from modules.positional_encoding import PositionalEncoding_Geo
from torch import nn
import torch
from modules.c_net import MLP  # Import MLP from c_net.py

# in dc_net there is nn to build Darcy's law equation.

# we hard constrain the equation
# including a isotropic/anisotropic permeability network k_net and a pressure network p_net. 

class K_Net(nn.Module):
    def __init__(self, neuron_num, hid_layer_num, positional_encode_nums, position_encode_freq_scale, anisotropic=True):
        super(K_Net, self).__init__()
        self.pos_encoder = PositionalEncoding_Geo(positional_encode_nums, position_encode_freq_scale)
        in_dim = sum([n*2 for n in positional_encode_nums]) + len(positional_encode_nums)
        out_dim = 3 if anisotropic else 1
        self.k_net = MLP(input_dim=in_dim, output_dim=out_dim, hidden_layers=hid_layer_num, hidden_features=neuron_num)
        self.anisotropic = anisotropic

    def forward(self, x):
        # x: (x,y,z)
        x_enc = self.pos_encoder(x)
        k = self.k_net(x_enc)
        if self.anisotropic:
            kx = torch.exp(k[:,0:1])
            ky = torch.exp(k[:,1:2])
            kz = torch.exp(k[:,2:3])
            return kx, ky, kz
        else:
            k_iso = torch.exp(k)
            return k_iso

# Pressure network, for steady state, input (x,y,z) and p not change by time.
class P_Net(nn.Module):
    def __init__(self, neuron_num, hid_layer_num, positional_encode_nums, position_encode_freq_scale):
        super(P_Net, self).__init__()
        self.pos_encoder = PositionalEncoding_Geo(positional_encode_nums, position_encode_freq_scale)
        in_dim = sum([n*2 for n in positional_encode_nums]) + len(positional_encode_nums)
        self.p_net = MLP(input_dim=in_dim, output_dim=1, hidden_layers=hid_layer_num, hidden_features=neuron_num)

    def forward(self, x):
        # x: (x,y,z)
        x_enc = self.pos_encoder(x)
        p = self.p_net(x_enc)
        return p

class DC_Net(nn.Module):
    # Darcy's law network, including k_net and p_net
    # and synthetic velocity calculation
    def __init__(self, neuron_num, hid_layer_num, positional_encode_nums, position_encode_freq_scale, anisotropic=True):
        super(DC_Net, self).__init__()
        self.k_net = K_Net(neuron_num, hid_layer_num, positional_encode_nums[:3], position_encode_freq_scale, anisotropic)
        self.p_net = P_Net(neuron_num, hid_layer_num, positional_encode_nums, position_encode_freq_scale)
        self.anisotropic = anisotropic
    
    def forward(self, x):
        x.requires_grad_(True)
        kx, ky, kz = self.k_net(x)
        p = self.p_net(x)
        # get velocity by Darcy's law
        grad_p = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[0]
        if self.anisotropic:
            kx, ky, kz = self.k_net(x)
            v = -torch.stack([grad_p[:, 0] / kx, grad_p[:, 1] / ky, grad_p[:, 2] / kz], dim=1)
        else:
            k = self.k_net(x)
            v = -grad_p / k
        return v