from modules.c_net import C_Net
from modules.data_module import DCEMRIDataModule
import torch
import numpy as np
from tqdm import tqdm

import os

def create_outlier_filter_mask(
    c_net: C_Net, 
    X_data, # This is a list [X_spatial, X_time]
    result_folder: str,
    batch_size=200_000,
):
    """
    Computes a boolean mask to filter out outlier data points based on gradient statistics.

    Args:
        c_net (C_Net): The pre-trained clean concentration network.
        X_data (list[torch.Tensor]): List containing spatial and temporal coordinates.
        batch_size (int): Batch size for processing to manage memory.

    Returns:
        torch.Tensor: A boolean tensor of shape (N,) where True means the point is valid.
    """
    log_file_path = os.path.join(result_folder, "training.log")
    os.makedirs(result_folder, exist_ok=True)
    def log_and_print(message):
        print(message)
        with open(log_file_path, 'a') as f:
            f.write(message + '\n')
    
    device = next(c_net.parameters()).device
    
    # Move all data to the device at once if it fits, otherwise batch
    X_spatial, X_time = X_data[0].to(device), X_data[1].to(device)

    # --- 1. Compute all gradients for the entire dataset ---
    all_grads = {
        'cx': [], 'cy': [], 'cz': [], 'pde_t_term': []
    }
    
    log_and_print("Filter Step 1: Computing pred_c and gradients for all data points...")
    pred_c_list = []
    for i in tqdm(range(0, X_spatial.shape[0], batch_size)):
        batch_X = X_spatial[i:i+batch_size]
        batch_t = X_time[i:i+batch_size]

        with torch.no_grad(): # We don't need gradients for the filter calculation itself
            pred_c = c_net.forward(torch.concat([batch_X, batch_t], dim=1))
            pred_c_list.append(pred_c.cpu())
            with torch.enable_grad(): # temporarily enable grad for input
                c_spatial_grad, c_t, c_diffusion = c_net.get_c_grad_ani_diffusion([batch_X, batch_t])
            # WARNING: c_diffusion already multiply by 1/Pe_g or diffusion coefficient
            c_spatial_grad = c_spatial_grad.detach()
            pde_t_term = c_t.detach() - c_diffusion.detach() # shaped (num_train_points,)
            all_grads['cx'].append(c_spatial_grad[:, 0].cpu())
            all_grads['cy'].append(c_spatial_grad[:, 1].cpu())
            all_grads['cz'].append(c_spatial_grad[:, 2].cpu())
            all_grads['pde_t_term'].append(pde_t_term.squeeze(-1).cpu())
    pred_c = torch.cat(pred_c_list).squeeze(-1) # shape (N,)
    # Concatenate all batch results
    for key in all_grads:
        all_grads[key] = torch.cat(all_grads[key])

    # --- 2. Calculate statistics per time step ---
    log_and_print("Filter Step 2: Calculating statistics for each time step...")
    unique_times = torch.unique(X_time.cpu())
    valid_mask = torch.ones(X_spatial.shape[0], dtype=torch.bool)

    for t_val in tqdm(unique_times):
        time_mask = (X_time.cpu() == t_val).squeeze() # masked all current timestep points, shaped (N,)
        discard_points_num = 0

        # first discard concentration value below 1e-1
        low_concentration_mask = (pred_c[time_mask] < 1e-1)
        discard_points_num += low_concentration_mask.sum().item()
        valid_mask[time_mask] &= ~low_concentration_mask

        # We process one gradient component at a time
        for grad_key in all_grads:
            grad_values_at_t = all_grads[grad_key][time_mask] # shaped (N,)
            
            # Important: Filter out near-zero values before taking the log
            # to avoid log(0) = -inf. A small epsilon is fine.
            significant_mask = torch.abs(grad_values_at_t) > 1e-8
            if not significant_mask.any():
                continue # Skip if no significant gradients at this time step
            # get log space.
            log_scaled_grads = torch.log(torch.abs(grad_values_at_t[significant_mask]))
            
            # Calculate mean and std for this component at this time
            mu_g = log_scaled_grads.mean()
            sigma_g = log_scaled_grads.std()
            
            # Define the valid range in log-space
            log_lower_bound = mu_g - 4 * sigma_g
            log_upper_bound = mu_g + 4 * sigma_g
            
            # Identify outliers ONLY among the significant gradients that were used for stats.
            outliers_in_significant = (
                (log_scaled_grads < log_lower_bound) |
                (log_scaled_grads > log_upper_bound)
            )
            
            # Initialize a mask for all points at this time step as False (not outliers)
            is_outlier_for_this_grad = torch.zeros_like(grad_values_at_t, dtype=torch.bool)
            
            # Place the results back into the full-size mask at the correct positions.
            is_outlier_for_this_grad[significant_mask] = outliers_in_significant
            
            # Update the global mask: if a point is an outlier for ANY gradient,
            # it becomes invalid (False).
            full_indices_at_t = torch.where(time_mask)[0] # shaped (cur_time_train_num_points,), store all indices
            outlier_indices = full_indices_at_t[is_outlier_for_this_grad]
            discard_points_num += valid_mask[outlier_indices].sum().item()
            # Mark these indices as invalid in the global mask
            valid_mask[outlier_indices] = False
        log_and_print(f"Time {t_val:.3f}: discard {discard_points_num} points ({discard_points_num/time_mask.sum().item():.2%})")
    num_filtered = (~valid_mask).sum().item()
    total_points = valid_mask.shape[0]
    log_and_print(f"Step 3: Filtering complete. Discarded {num_filtered}/{total_points} points ({num_filtered/total_points:.2%}).")

    return valid_mask


# Assuming DCEMRIDataModule is your previous datamodule
# We can inherit from it to reuse some logic
class FilteredDCEMRIDataModule(DCEMRIDataModule):
    def __init__(self, original_datamodule, filter_mask):
        """
        Initializes with a pre-existing datamodule and a filter mask.
        """
        # Copy essential attributes
        super().__init__(
            data=original_datamodule.data * original_datamodule.C_star, # un-normalize then re-normalize
            char_domain=original_datamodule.char_domain,
            batch_size=original_datamodule.batch_size
        )
        self.original_datamodule = original_datamodule
        self.filter_mask = filter_mask # save the mask for visualization

    def setup(self, stage=None):
        # First, run the original setup to get self.X_train, self.C_train etc.
        self.original_datamodule.setup()
        
        # Now, apply the filter mask to the training data
        # print("Applying pre-computed filter mask to the training dataset...")
        
        # Ensure the mask is the correct size
        assert self.filter_mask.shape[0] == self.original_datamodule.X_train[0].shape[0], \
            "Filter mask size does not match training data size!"

        self.X_train = [
            self.original_datamodule.X_train[0][self.filter_mask],
            self.original_datamodule.X_train[1][self.filter_mask]
        ]
        self.C_train = self.original_datamodule.C_train[self.filter_mask]
        
        # The indices for RBA are now relative to the NEW, smaller dataset
        self.num_train_points = self.X_train[0].shape[0]
        self.X_train_indice = torch.arange(self.num_train_points, dtype=torch.long)
        
        # We don't filter the validation set
        self.X_val = self.original_datamodule.X_val
        self.C_val = self.original_datamodule.C_val
        self.X_val_indice = self.original_datamodule.X_val_indice

        # print(f"Filtered training set now has {self.num_train_points} points.")

    # The train_dataloader and val_dataloader methods can be inherited or reused
    # from your base RBAResampleDataModule class, as they will now operate on the
    # new, smaller self.X_train tensors.
