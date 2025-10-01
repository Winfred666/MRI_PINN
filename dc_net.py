import numpy as np
from positional_encoding import PositionalEncoding_Geo, PositionalEncoding_GeoTime
from torch import nn
import torch
# in dc_net there is nn to build Darcy's law equation.

# we hard constrain the equation
# including a isotropic/anisotropic permeability network k_net and a pressure network p_net. 
