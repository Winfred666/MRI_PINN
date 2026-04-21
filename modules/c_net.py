from modules.data_module import CharacteristicDomain
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.func import grad, jacrev, vmap
from torch.utils.checkpoint import checkpoint_sequential

from modules.positional_encoding import PositionalEncoding_GeoTime
from utils.visualize import visualize_prediction_vs_groundtruth

# Containing basic concentration network, 
# and denoised concentration network (2 sub network)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layers,
        hidden_features,
        use_activation_checkpointing=True,
        checkpoint_segments=4,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_activation_checkpointing = bool(use_activation_checkpointing)
        self.checkpoint_segments = max(1, int(checkpoint_segments))
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
        if self.use_activation_checkpointing and self.training and torch.is_grad_enabled():
            num_layers = len(self.net)
            if num_layers > 2:
                num_segments = min(self.checkpoint_segments, num_layers)
                return checkpoint_sequential(self.net, num_segments, x, use_reentrant=False)
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
        self.C_star = float(np.asarray(C_star).reshape(-1)[0])

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

        # --- Denoise Module ---
        self.noise_mlp = MLP(
            input_dim=c_input_dim,
            output_dim=1,
            hidden_layers=5,  # from Table D.8
            hidden_features=66
        )
        self.sigma_reg_weight = 0.001 # Regularization weight for sigma
        self.sigma_0 = 0.01  # minimum noise level

        # Validation uses five interpolated time points across the full sequence
        # but always visualizes only the mid-z slice in physical concentration units.
        self.val_slice_zindex = [int(char_domain.domain_shape[2] // 2)]
        self.val_slice_t_normalized = np.linspace(
            float(char_domain.t_normalized[0]),
            float(char_domain.t_normalized[-1]),
            5,
            dtype=np.float32,
        )
        self.val_slice_4d = self._build_validation_grid()
        self.gt_data = np.asarray(data, dtype=np.float32)
        self.val_slice_gt_c = self._build_validation_gt_slices()

    def _build_validation_grid(self):
        nx, ny, nz = self.char_domain.domain_shape[:3]
        x = np.linspace(-1.0, 1.0, nx, dtype=np.float32)
        y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
        z = np.linspace(-1.0, 1.0, nz, dtype=np.float32)[self.val_slice_zindex]
        X, Y, Z, T = np.meshgrid(
            x,
            y,
            z,
            self.val_slice_t_normalized,
            indexing="ij",
        )
        grid_points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel(), T.ravel()])
        return torch.as_tensor(grid_points, dtype=torch.float32, device=self.char_domain.device)

    def _interpolate_gt_volume(self, normalized_time):
        target_time = float(self.char_domain.recover_time_numpy(np.asarray(normalized_time, dtype=np.float32)))
        time_axis = np.asarray(self.char_domain.t, dtype=np.float32)

        if target_time <= float(time_axis[0]):
            return self.gt_data[:, :, :, 0]
        if target_time >= float(time_axis[-1]):
            return self.gt_data[:, :, :, -1]

        upper = int(np.searchsorted(time_axis, target_time, side="right"))
        lower = max(0, upper - 1)
        upper = min(upper, len(time_axis) - 1)
        t0 = float(time_axis[lower])
        t1 = float(time_axis[upper])
        if upper == lower or t1 <= t0:
            return self.gt_data[:, :, :, lower]

        alpha = (target_time - t0) / (t1 - t0)
        return (1.0 - alpha) * self.gt_data[:, :, :, lower] + alpha * self.gt_data[:, :, :, upper]

    def _build_validation_gt_slices(self):
        gt_slices = []
        z_index = self.val_slice_zindex[0]
        for normalized_time in self.val_slice_t_normalized:
            interpolated_volume = self._interpolate_gt_volume(normalized_time)
            gt_slices.append(interpolated_volume[:, :, z_index])
        return np.stack(gt_slices, axis=-1)[:, :, None, :]

    def forward(self, X_train):
        # Explicitly define the forward pass logic for concentration
        if self.c_pos_encoder:
            encoded_X = self.c_pos_encoder(X_train)
        else:
            encoded_X = X_train
        return self.c_mlp(encoded_X)

    def sigma_forward(self, X_train):
        # Forward pass for the noise prediction
        if self.c_pos_encoder:
            txyz_freq = self.c_pos_encoder(X_train)
        else:
            txyz_freq = X_train
        predicted_sigma = 10.0 * torch.sigmoid(self.noise_mlp(txyz_freq)) + self.sigma_0
        return predicted_sigma

    # Here the batch is c_dataset training point.
    def calculate_denoise_loss(self, batch):
        Xt, _, c_observed = batch
        X_full = torch.cat(Xt, dim=1)
        c_clean_pred = self.forward(X_full)
        sigma_pred = self.sigma_forward(X_full)
        # Negative log likelihood per-point
        errp2 = (c_observed - c_clean_pred) ** 2
        eps = 1e-8
        log_term = torch.log((sigma_pred / self.sigma_0) ** 2 + eps) / 2
        quad_term = errp2 / (2.0 * (sigma_pred ** 2 + eps))
        nll_loss = log_term + quad_term

        if self.sigma_reg_weight > 0:
            target = errp2 + eps
            reg = (torch.log(sigma_pred ** 2 + eps) - torch.log(target)) ** 2
            nll_loss = nll_loss + self.sigma_reg_weight * reg

        return nll_loss, sigma_pred, errp2

    def draw_concentration_slices(self, include_error=True):
        with torch.no_grad():
            # Ensure val_slice_4d is on the correct device (vanilla pytorch has to do so)
            device = next(self.parameters()).device
            val_coords = self.val_slice_4d.to(device)
            
            vol_disp_all = self(val_coords).cpu().numpy().reshape(
                self.char_domain.domain_shape[0],
                self.char_domain.domain_shape[1],
                len(self.val_slice_zindex),
                len(self.val_slice_t_normalized),
            )
            vol_disp_all = vol_disp_all * self.C_star
            c_vis_list = []
            for i in range(len(self.val_slice_zindex)):
                for j in range(len(self.val_slice_t_normalized)):
                    slice_gt_c = self.val_slice_gt_c[:, :, i, j]
                    vol_disp = vol_disp_all[:, :, i, j]
                    vol_disp *= self.char_domain.mask[:, :, self.val_slice_zindex[i]]
                    c_vis_list.append(
                        visualize_prediction_vs_groundtruth(
                            vol_disp,
                            slice_gt_c,
                            include_error=include_error,
                        )
                    )
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
    def get_c_grad_ani_diffusion(self, Xt, learnable_D_factor=None):
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
        is_anisotropic = (self.char_domain.DTI_or_coef.ndim != 0)
        if is_anisotropic:
            grid = X.view(1, -1, 1, 1, 3) # shape (1, N, 1, 1, 3) for sampling
            D_tensor = F.grid_sample(self.char_domain.DTI_or_coef, grid, mode='nearest', align_corners=True)
            D_tensor = D_tensor.view(-1, 3, 3) # shape (N, 3, 3)
            diff_flux = torch.bmm(D_tensor, c_spatial_grad.unsqueeze(-1)).squeeze(-1)
        else:
            diff_flux = c_spatial_grad

        # 4. Compute the Laplacian (divergence of the flux) component-wise.
        # This is the most memory-efficient way.
        c_laplacian = torch.zeros_like(t).squeeze(-1)
        for i in range(3):
            _retain_graph = (learnable_D_factor is not None or (i < 2))
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
        
        # 5. Scale when isotropic or we has learnable D factor.
        if learnable_D_factor is not None:
            c_laplacian = learnable_D_factor * c_laplacian
        elif not is_anisotropic: # when learnable_D_factor is not accessable, use default char_domain.DTI_or_coef
            c_laplacian = self.char_domain.DTI_or_coef * c_laplacian
        return c_spatial_grad, c_t, c_laplacian.unsqueeze(1) # shaped (N, 1)

    # calculate gradient and diffusion term outside (better getting anisotropic diffusion beforehand)
    def get_TD_RBA_scale(self, t_list, c_grad, c_diffusion):
        assert c_grad is not None and c_diffusion is not None
        # detach early to avoid extra gradient flow
        t_list = t_list.detach()
        c_grad = c_grad.detach()
        c_diffusion = c_diffusion.detach()
        # Calculate the scaling factor for each time point
        all_terms = torch.stack([
            c_grad[:, 0].abs(),
            c_grad[:, 1].abs(),
            c_grad[:, 2].abs(),
            # c_diffusion is already the characteristic diffusion term.
            (c_grad[:, 3] - c_diffusion.squeeze()).abs()
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
