from modules.data_module import CharacteristicDomain
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.func import grad, jacrev, vmap

from modules.positional_encoding import PositionalEncoding_GeoTime
from utils.visualize import visualize_prediction_vs_groundtruth

# Containing basic concentration network, 
# and denoised concentration network (2 sub network)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers, hidden_features):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers = [nn.Linear(input_dim, hidden_features), nn.SiLU()]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_features, hidden_features))
            # WARNING: use SiLU instead of tanh to avoid vanishing gradient
            layers.append(nn.SiLU())
        layers.append(nn.Linear(hidden_features, output_dim))
        self.net = nn.Sequential(*layers)

        # initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class C_Net(nn.Module):
    # data is only for validation visualization
    def __init__(
        self,
        c_layers,
        data, char_domain: CharacteristicDomain, C_star,
        positional_encoding=True,
        freq_nums=(8, 8, 8, 0),
        gamma_space=1.0,
    ):
        super().__init__()
        self.c_layers = c_layers
        self.positional_encoding = positional_encoding
        self.char_domain = char_domain

        c_input_dim = 4
        if positional_encoding:
            num_freq_space = np.array(freq_nums[:3])
            num_freq_time = freq_nums[3]
            self.c_pos_encoder = PositionalEncoding_GeoTime(
                num_freq_space,
                num_freq_time,
                include_input=True,
                gamma_space=gamma_space,
            )
            # Correctly calculate the input dimension for the MLP
            c_input_dim = 4 + int(num_freq_space.sum()) * 2 + int(num_freq_time) * 2
        else:
            self.c_pos_encoder = None

        # Define the MLP as a separate attribute
        self.c_mlp = MLP(
            input_dim=c_input_dim,
            output_dim=c_layers[-1],
            hidden_layers=len(c_layers) - 2,
            hidden_features=c_layers[1],
        )

        # --- Keep validation attributes ---
        self.val_slice_z = [char_domain.domain_shape[2] // 2 - 6, char_domain.domain_shape[2] // 2, char_domain.domain_shape[2] // 2 + 6]
        base_t = char_domain.domain_shape[3] // 4 * 2
        self.val_slice_t = [base_t - 6, base_t - 3, base_t, base_t + 3]
        self.val_slice_4d = char_domain.get_characteristic_geotimedomain(slice_zindex=self.val_slice_z,
                                                                       slice_tindex=self.val_slice_t) # (?,4)
        Z, T = np.meshgrid(self.val_slice_z, self.val_slice_t, indexing='ij')
        self.val_slice_gt_c = data[:, :, Z, T] / C_star

    def forward(self, X_train):
        # Explicitly define the forward pass logic
        if self.c_pos_encoder:
            x = self.c_pos_encoder(X_train)
        return self.c_mlp(x)

    def draw_concentration_slices(self, ):
        with torch.no_grad():
            # Ensure val_slice_4d is on the correct device (vanilla pytorch has to do so)
            device = next(self.parameters()).device
            val_coords = self.val_slice_4d.to(device)
            
            vol_disp_all = self(val_coords).cpu().numpy().reshape(
                self.char_domain.domain_shape[0], self.char_domain.domain_shape[1], len(self.val_slice_z), len(self.val_slice_t))
            c_vis_list = []
            for i in range(len(self.val_slice_z)):
                for j in range(len(self.val_slice_t)):
                    slice_gt_c = self.val_slice_gt_c[:, :, i, j]
                    vol_disp = vol_disp_all[:, :, i, j]
                    vol_disp *= self.char_domain.mask[:, :, self.val_slice_z[i]]
                    c_vis_list.append(visualize_prediction_vs_groundtruth(vol_disp, slice_gt_c))
            return np.hstack(c_vis_list)
    
    # X should be the raw tuple extract from datamodule. D_coef could be number or tensor of shape (N,3,3)
    # using vmap + jacrev to calculate the laplacian, will consume more memory but may be faster
    def get_c_grad_ani_diffusion_full_jcb(self, Xt): # here X is a tuple of (x,y,z,t)
        X, t = Xt
        X.requires_grad_(True) # Still need this for the final backprop
        t.requires_grad_(True)
        # 1. Define a function that computes c for a SINGLE sample
        def get_c_single(X, t):
            # x is (3,), t is (1,), input must be (1,4,)
            return self(torch.cat([X, t]).unsqueeze(0)).squeeze()

        # 2. Get the gradient w.r.t. spatial coords (X) and time (t) for the whole batch
        # use vmap to apply the grad function to each sample in the batch
        c_spatial_grad, c_t = vmap(grad(get_c_single, argnums=(0, 1)))(X, t)
        
        # 3. Define a function to compute the diffusive flux for a SINGLE sample
        def get_flux_single(x, t,D_tensor_for_sample, is_anisotropic):
            # We need to re-evaluate the model to build the graph for the jacobian
            def c_from_x(x_inner):
                return get_c_single(x_inner, t)
            c_grad_spatial_at_x = grad(c_from_x)(x)
            if is_anisotropic: # Anisotropic
                flux = D_tensor_for_sample @ c_grad_spatial_at_x
            else:  # Isotropic
                flux = c_grad_spatial_at_x  # We will scale by D later
            return flux

        # 4. Prepare the arguments for the vmap call.
        is_anisotropic = (self.char_domain.DTI_or_coef.ndim != 0)
        if is_anisotropic:
            # Pre-calculate the entire batch of D tensors.
            grid = X.view(1, -1, 1, 1, 3)
            D_tensor_batch = F.grid_sample(self.char_domain.DTI_grid, grid, mode='nearest', align_corners=True)
            D_tensor_batch = D_tensor_batch.view(-1, 3, 3)  # Shape: (N, 3, 3)
        else:
            # Create a placeholder if not needed. Its values won't be used.
            D_tensor_batch = torch.empty(X.shape[0], 3, 3, device=X.device)

        # 4. Compute the Jacobian of the flux function for each sample and get its trace (the Laplacian)
        # jacrev(get_flux_single) gives the full 3x3 Jacobian matrix, Differentiate the pure function w.r.t. x (arg 0).
        # We vmap this over the batch.
        flux_jacobians = vmap(jacrev(get_flux_single, argnums=0), in_dims=(0, 0, 0, None))(X, t, D_tensor_batch, is_anisotropic)
        c_laplacian = torch.diagonal(flux_jacobians, offset=0, dim1=-2, dim2=-1).sum(-1)

        # For isotropic case, scale the final result
        if self.char_domain.DTI_or_coef.ndim == 0:
            c_laplacian = self.char_domain.DTI_or_coef * c_laplacian

        return c_spatial_grad, c_t, c_laplacian.unsqueeze(1)

    # retain_graph if need backprop again like pde loss; provided_DTI_or_coef could be learnable.
    def get_c_grad_ani_diffusion(self, Xt, learnable_DTI_or_coef=None):
        X, t = Xt
        X.requires_grad_(True)
        t.requires_grad_(True)
        # 1. Get the model's prediction for the whole batch.
        pred_c = self(torch.cat(Xt, dim=-1))
        # 2. Compute the first-order spatial gradient for the whole batch.
        c_spatial_grad, c_t = torch.autograd.grad(
            pred_c,
            (X, t),
            grad_outputs=torch.ones_like(pred_c),
            create_graph=True,
        )
        # 3. Calculate the diffusive flux for the whole batch.
        DTI_or_coef = self.char_domain.DTI_or_coef if learnable_DTI_or_coef is None else learnable_DTI_or_coef
        is_anisotropic = (DTI_or_coef.ndim != 0)
        if is_anisotropic:
            grid = X.view(1, -1, 1, 1, 3)
            D_tensor = F.grid_sample(DTI_or_coef, grid, mode='nearest', align_corners=True)
            D_tensor = D_tensor.view(-1, 3, 3)
            diff_flux = torch.bmm(D_tensor, c_spatial_grad.unsqueeze(-1)).squeeze(-1)
        else:
            diff_flux = c_spatial_grad

        # 4. Compute the Laplacian (divergence of the flux) component-wise.
        # This is the most memory-efficient way.
        c_laplacian = torch.zeros_like(t).squeeze(-1)
        for i in range(3):
            _retain_graph = (learnable_DTI_or_coef is not None or (i < 2))
            # We compute the gradient of the i-th component of the flux w.r.t. full X.
            # We do NOT need create_graph=True here, as we don't differentiate the Laplacian itself.
            grad_of_flux_comp_i, = torch.autograd.grad(
                outputs=diff_flux[:, i],
                inputs=X,
                grad_outputs=torch.ones_like(diff_flux[:, i]),
                retain_graph=_retain_graph,
            )
            # We only keep the i-th component of the result (the diagonal term).
            c_laplacian += grad_of_flux_comp_i[:, i]
        # 5. Scale for the isotropic case.
        if not is_anisotropic:
            c_laplacian = DTI_or_coef * c_laplacian
        return c_spatial_grad, c_t, c_laplacian.unsqueeze(1) # shaped (N, 1)

    # calculate gradient and laplace outside (better getting anisotropic c_laplacian beforehand)
    def get_TD_RBA_scale(self, t_list, c_grad, c_laplacian):
        assert c_grad is not None and c_laplacian is not None
        # detach early to avoid extra gradient flow
        t_list = t_list.detach()
        c_grad = c_grad.detach()
        c_laplacian = c_laplacian.detach()
        # Calculate the scaling factor for each time point
        all_terms = torch.stack([
            c_grad[:, 0].abs(),
            c_grad[:, 1].abs(),
            c_grad[:, 2].abs(),
            (c_grad[:, 3] - (1/self.char_domain.Pe_g) * c_laplacian.squeeze()).abs()
        ], dim=1) # Shape: (N, 4)

        # Vectorized approach to find max scale per unique time
        unique_t, inverse_indices = torch.unique(t_list.squeeze(), return_inverse=True)
        
        # Use scatter_reduce_ to find the max of all terms for each unique time
        # We need to find the max across all points that share a time value
        max_terms_per_point = torch.max(all_terms, dim=1).values # Shape: (N,)
        
        # Then, find the max among points with the same time (T,)
        max_scales_per_unique_t = torch.zeros_like(unique_t, dtype=max_terms_per_point.dtype)
        max_scales_per_unique_t.scatter_reduce_(0, inverse_indices, max_terms_per_point, reduce="amax")

        # Broadcast the max scale back to all points (N,), need to detach.
        t_scale = max_scales_per_unique_t[inverse_indices].detach()
        
        return t_scale # Return shape (N,) to match pointwise_loss
