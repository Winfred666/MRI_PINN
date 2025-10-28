import numpy as np
from scipy.ndimage import map_coordinates
import torch
import torch.nn.functional as F

# --- PyTorch configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def interpolate_tricubic_pytorch(volume, points):
    """
    Performs tricubic interpolation on a 3D volume at specified points.
    This is a custom implementation since PyTorch's grid_sample doesn't support
    bicubic mode for 3D data.
    
    Args:
        volume (torch.Tensor): The 3D volume of shape (nx, ny, nz).
        points (torch.Tensor): The points to interpolate at, shape (..., 3).
                               Coordinates are in voxel space.
    Returns:
        torch.Tensor: The interpolated values, shape (...).
    """
    nx, ny, nz = volume.shape
    
    # Get the integer and fractional parts of the coordinates
    p_floor = torch.floor(points).long()
    p_frac = points - p_floor.float()

    # --- Catmull-Rom cubic interpolation kernel ---
    def cubic_weights(frac):
        w0 = -0.5 * frac + frac**2 - 0.5 * frac**3
        w1 = 1.0 - 2.5 * frac**2 + 1.5 * frac**3
        w2 = 0.5 * frac + 2.0 * frac**2 - 1.5 * frac**3
        w3 = -0.5 * frac**2 + 0.5 * frac**3
        return torch.stack([w0, w1, w2, w3], dim=-1)

    # --- Gather data from the 4x4x4 neighborhood ---
    # Create tensors to hold the 64 neighbor values for each point
    neighbors = torch.zeros(points.shape[:-1] + (4, 4, 4), device=device, dtype=torch.float32)

    for i in range(4):
        for j in range(4):
            for k in range(4):
                # Get coordinates for the neighbor (i, j, k)
                x_coords = torch.clamp(p_floor[..., 0] - 1 + i, 0, nx - 1)
                y_coords = torch.clamp(p_floor[..., 1] - 1 + j, 0, ny - 1)
                z_coords = torch.clamp(p_floor[..., 2] - 1 + k, 0, nz - 1)
                
                # Gather the value from the volume
                neighbors[..., i, j, k] = volume[x_coords, y_coords, z_coords]

    # --- Interpolate along each axis ---
    # 1. Interpolate along Z
    w_z = cubic_weights(p_frac[..., 2])
    # ...ijk are the neighborhood dims, ...k are the weights. Sum over k.
    interp_z = torch.einsum('...ijk,...k->...ij', neighbors, w_z)
    # 2. Interpolate along Y
    w_y = cubic_weights(p_frac[..., 1])
    # ...ij are the interpolated planes, ...j are the weights. Sum over j.
    interp_y = torch.einsum('...ij,...j->...i', interp_z, w_y)
    # 3. Interpolate along X
    w_x = cubic_weights(p_frac[..., 0])
    # ...i are the interpolated lines, ...i are the weights. Sum over i.
    final_interp = torch.einsum('...i,...i->...', interp_y, w_x)
    return final_interp


def advection_step_pytorch(C, vx, vy, vz, dt):
    """
    Performs one step of advection using PyTorch for GPU acceleration.
    This implementation uses RK4 for temporal integration and custom tricubic interpolation.
    """
    nx, ny, nz = C.shape
    
    # Create a grid of destination coordinates
    x_coords, y_coords, z_coords = torch.meshgrid(
        torch.arange(nx, device=device, dtype=torch.float32),
        torch.arange(ny, device=device, dtype=torch.float32),
        torch.arange(nz, device=device, dtype=torch.float32),
        indexing='ij'
    )
    p = torch.stack([x_coords, y_coords, z_coords], dim=-1)

    # --- Helper for interpolation (using bilinear for speed on velocity) ---
    def interpolate_velocity(v, points):
        v_tensor = v.view(1, 1, nx, ny, nz)
        normalized_points = 2.0 * points / torch.tensor([nx-1, ny-1, nz-1], device=device) - 1.0
        grid = normalized_points.flip(dims=(-1,)).view(1, nx, ny, nz, 3)
        return F.grid_sample(v_tensor, grid, mode='bilinear', padding_mode='border', align_corners=True).squeeze()

    # --- RK4 Back-tracing ---
    h = dt
    k1 = torch.stack([interpolate_velocity(vx, p), interpolate_velocity(vy, p), interpolate_velocity(vz, p)], dim=-1)
    p_k2 = p - 0.5 * h * k1
    k2 = torch.stack([interpolate_velocity(vx, p_k2), interpolate_velocity(vy, p_k2), interpolate_velocity(vz, p_k2)], dim=-1)
    p_k3 = p - 0.5 * h * k2
    k3 = torch.stack([interpolate_velocity(vx, p_k3), interpolate_velocity(vy, p_k3), interpolate_velocity(vz, p_k3)], dim=-1)
    p_k4 = p - h * k3
    k4 = torch.stack([interpolate_velocity(vx, p_k4), interpolate_velocity(vy, p_k4), interpolate_velocity(vz, p_k4)], dim=-1)
    
    departure_points = p - (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    # --- Interpolate concentration at departure points using custom tricubic---
    min_C, max_C = C.min(), C.max()
    
    # C_advected = F.grid_sample(
    #     C.view(1, 1, nx, ny, nz),
    #     (2.0 * departure_points / torch.tensor([nx-1, ny-1, nz-1], device=device) - 1.0).flip(dims=(-1,)).view(1, nx, ny, nz, 3),
    #     mode='bilinear',
    #     padding_mode='border',
    #     align_corners=True
    # ).squeeze()
    C_advected = interpolate_tricubic_pytorch(C, departure_points)

    # Clip to prevent overshooting, which can happen with cubic interpolation
    return torch.clamp(C_advected, min_C, max_C)


def gradient_torch(f, dx, dy, dz):
    """Computes the gradient of a 3D tensor using central differences."""
    grad_x = (torch.roll(f, -1, dims=0) - torch.roll(f, 1, dims=0)) / (2 * dx)
    grad_y = (torch.roll(f, -1, dims=1) - torch.roll(f, 1, dims=1)) / (2 * dy)
    grad_z = (torch.roll(f, -1, dims=2) - torch.roll(f, 1, dims=2)) / (2 * dz)
    # Handle boundaries with forward/backward differences
    grad_x[0, :, :] = (f[1, :, :] - f[0, :, :]) / dx
    grad_x[-1, :, :] = (f[-1, :, :] - f[-2, :, :]) / dx
    grad_y[:, 0, :] = (f[:, 1, :] - f[:, 0, :]) / dy
    grad_y[:, -1, :] = (f[:, -1, :] - f[:, -2, :]) / dy
    grad_z[:, :, 0] = (f[:, :, 1] - f[:, :, 0]) / dz
    grad_z[:, :, -1] = (f[:, :, -1] - f[:, :, -2]) / dz
    return grad_x, grad_y, grad_z

def divergence_torch(fx, fy, fz, dx, dy, dz):
    """Computes the divergence of a 3D vector field."""
    dfx_dx, _, _ = gradient_torch(fx, dx, dy, dz)
    _, dfy_dy, _ = gradient_torch(fy, dx, dy, dz)
    _, _, dfz_dz = gradient_torch(fz, dx, dy, dz)
    return dfx_dx + dfy_dy + dfz_dz

def laplacian_torch(f, dx, dy, dz):
    """Computes the Laplacian of a 3D tensor."""
    grad_x, grad_y, grad_z = gradient_torch(f, dx, dy, dz)
    return divergence_torch(grad_x, grad_y, grad_z, dx, dy, dz)


def diffusion_step_pytorch(C, D, dt, dx, dy, dz, num_sub_steps=5):
    """Performs one step of diffusion using PyTorch for GPU acceleration."""
    sub_dt = dt / num_sub_steps
    C_new = C.clone()

    for _ in range(num_sub_steps):
        if isinstance(D, (float, int)):
            # Case 1: D is a constant scalar
            laplacian_C = laplacian_torch(C_new, dx, dy, dz)
            C_new += sub_dt * D * laplacian_C
        
        elif D.ndim == 5: # D is a tensor field
            # 1. Compute gradient of concentration ∇C
            grad_C_x, grad_C_y, grad_C_z = gradient_torch(C_new, dx, dy, dz)
            grad_C_vec = torch.stack([grad_C_x, grad_C_y, grad_C_z], dim=-1)

            # 2. Compute flux J = D @ ∇C
            flux = torch.einsum('...ij,...j->...i', D, grad_C_vec)
            
            # 3. Compute divergence of the flux vector ∇·J
            divergence_of_flux = divergence_torch(flux[..., 0], flux[..., 1], flux[..., 2], dx, dy, dz)
            C_new += sub_dt * divergence_of_flux
        elif D.ndim == 0: # D is a scalar tensor
            laplacian_C = laplacian_torch(C_new, dx, dy, dz)
            C_new += sub_dt * D.item() * laplacian_C
        else:
            raise ValueError(f"Unsupported shape for diffusion coefficient D: {D.shape}")
            
    return C_new


def advect_diffuse_forward_simulation(
    C_initial, vx, vy, vz, D, 
    total_time, num_steps=10, 
    voxel_dims=(0.1, 0.1, 0.1),
    use_gpu=True
):
    """
    Runs a full forward simulation using operator splitting.
    Can use NumPy (use_gpu=False) or PyTorch (use_gpu=True).
    """
    if use_gpu and device.type == 'cpu':
        print("Warning: GPU not available, falling back to CPU. PyTorch simulation may be slow.")
    
    dt = total_time / num_steps
    dx, dy, dz = voxel_dims

    if use_gpu:
        # --- PyTorch Simulation ---
        C_current = torch.from_numpy(C_initial.copy()).to(device, dtype=torch.float32)
        vx_t = torch.from_numpy(vx).to(device, dtype=torch.float32)
        vy_t = torch.from_numpy(vy).to(device, dtype=torch.float32)
        vz_t = torch.from_numpy(vz).to(device, dtype=torch.float32)
        D_t = torch.from_numpy(D).to(device, dtype=torch.float32) if isinstance(D, np.ndarray) else D

        vx_vox_dt = vx_t / dx * dt
        vy_vox_dt = vy_t / dy * dt
        vz_vox_dt = vz_t / dz * dt
        
        simulation_frames = [C_current.cpu().numpy()]

        print(f"Running PyTorch simulation on {device} for {total_time}s in {num_steps} steps (dt={dt:.3f}s)")
        for i in range(num_steps):
            C_advected = advection_step_pytorch(C_current, vx_vox_dt, vy_vox_dt, vz_vox_dt, dt=1.0)
            C_diffused = diffusion_step_pytorch(C_advected, D_t, dt, dx, dy, dz)
            C_current = C_diffused
            simulation_frames.append(C_current.cpu().numpy())
    print("Simulation complete.")
    return simulation_frames


def generate_block_concentration(nx, ny, nz, noise_level=0.0, center=(0.5,0.5,0.5), size=(0.2,0.2,0.5)):
    """Generates a central block of concentration for demonstration."""
    C_initial = np.zeros((nx, ny, nz))
    half_size_x = max(1, int(nx * size[0] / 2))
    half_size_y = max(1, int(ny * size[1] / 2))
    half_size_z = max(1, int(nz * size[2] / 2))
    cx, cy, cz = int(nx * center[0]), int(ny * center[1]), int(nz * center[2])
    x0, x1 = max(0, cx - half_size_x), min(nx, cx + half_size_x)
    y0, y1 = max(0, cy - half_size_y), min(ny, cy + half_size_y)
    z0, z1 = max(0, cz - half_size_z), min(nz, cz + half_size_z)
    C_initial[x0:x1, y0:y1, z0:z1] = 100.0
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, C_initial.shape)
        C_initial += noise
    return C_initial

def generate_swirl_velocity_field(nx, ny, nz, voxel_size_mm, v0=0.03):
    """Generates a swirl velocity field around the z-axis for demonstration."""
    velocity_field_u = np.zeros((nx, ny, nz), dtype=float)
    velocity_field_v = np.zeros((nx, ny, nz), dtype=float)
    velocity_field_w = np.zeros((nx, ny, nz), dtype=float)

    # Center of rotation in physical units (mm)
    center_x_mm = (nx - 1) * voxel_size_mm / 2.0
    center_y_mm = (ny - 1) * voxel_size_mm / 2.0

    for ix in range(nx):
        x_mm = ix * voxel_size_mm
        for iy in range(ny):
            y_mm = iy * voxel_size_mm
            dx_mm = x_mm - center_x_mm
            dy_mm = y_mm - center_y_mm
            r = np.hypot(dx_mm, dy_mm)

            if r > 1e-12:
                # Velocity magnitude increases with the square of the radius
                ux = -v0 * (r**2) * (dy_mm / r)
                vy =  v0 * (r**2) * (dx_mm / r)
            else:
                ux, vy = 0.0, 0.0
            
            # Apply the same 2D velocity to all z-slices
            velocity_field_u[ix, iy, :] = ux
            velocity_field_v[ix, iy, :] = vy
            # velocity_field_w remains 0.0

    return velocity_field_u, velocity_field_v, velocity_field_w


# v_cell_val is about how many cell move per time unit(minute). 1 means 1 voxel per minute
def generate_constant_velocity_field(nx, ny, nz, voxel_size_mm, v_cell_val=(1.0,1.0,1.0)):
    """Generates a constant velocity field for demonstration."""
    velocity_field_u = np.full((nx, ny, nz), v_cell_val[0] * voxel_size_mm, dtype=float)
    velocity_field_v = np.full((nx, ny, nz), v_cell_val[1] * voxel_size_mm, dtype=float)
    velocity_field_w = np.full((nx, ny, nz), v_cell_val[2] * voxel_size_mm, dtype=float)
    return velocity_field_u, velocity_field_v, velocity_field_w

# could also generate D_coeff as a tensor field if needed
def generate_diffusion_coefficient_field(nx, ny, nz, voxel_size_mm, D_eigvecs, D_eigvals):
    """Generates a tensor diffusion coefficient field."""
    D_tensor = np.zeros((nx, ny, nz, 3, 3), dtype=float)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                D_tensor[i, j, k] = D_eigvecs @ np.diag(D_eigvals) @ D_eigvecs.T
    return D_tensor

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # 1. Setup simulation parameters
    voxel_size_mm = 0.1 # mm
    dt_mri = 60 # seconds between MRI frames
    nx, ny, nz = (64, 64, 32)

    # 2. Generate initial data
    
    # 2.1 D_coeff: Diffusion coefficient (in mm^2/s)
    # D_coeff = 1e-3  # mm^2/s, constant
    # Set D_eigen vectors as a transformed matrix
    D_eigvals = np.array([0.0005, 0.0001, 0.00001])  # mm^2/s
    D_eigvecs = np.array([
        [np.cos(np.pi * 10), -np.sin(np.pi * 10), 0],
        [np.sin(np.pi / 4),  np.cos(np.pi / 4), 0],
        [0, 0, 1]
    ]).T
    D_coeff = generate_diffusion_coefficient_field(
        nx, ny, nz, voxel_size_mm, D_eigvecs, D_eigvals
    )
    
    # 2.2 C_observed_t0: The 3D concentration frame at time t=0 (your starting point)
    C_observed_t0 = generate_block_concentration(nx, ny, nz)
    
    # 2.3 velocity_field_u,v,w: The 3D velocity fields (in mm/s)
    # velocity_field_u, velocity_field_v, velocity_field_w = generate_swirl_velocity_field(
    #     nx, ny, nz, voxel_size_mm, v0=0.03
    # )
    velocity_field_u, velocity_field_v, velocity_field_w = generate_constant_velocity_field(
        nx, ny, nz, voxel_size_mm, v_cell_val=(0.1, 0.1, 0.0)
    )


    # 3. Run the simulation from one MRI frame to the next
    # Use a high number of sub-steps for accuracy
    simulation_results = advect_diffuse_forward_simulation(
        C_initial=C_observed_t0,
        vx=velocity_field_u,
        vy=velocity_field_v,
        vz=velocity_field_w,
        D=D_coeff,
        total_time=dt_mri,
        num_steps=100, # More sub-steps lead to higher accuracy, but slower.
        voxel_dims=(voxel_size_mm, voxel_size_mm, voxel_size_mm),
        use_gpu=True # Set to True to use PyTorch and GPU
    )

    # The final frame of the simulation is your prediction for the next MRI frame
    C_predicted_t1 = simulation_results[-1]

    # 4. Visualize the results
    import matplotlib.pyplot as plt

    mid_slice_idx = nz // 2

    n_frames = len(simulation_results)
    n_show = 6
    idxs = np.linspace(0, n_frames - 1, n_show, dtype=int)
    dt_sub = dt_mri / (n_frames - 1) if n_frames > 1 else 0.0

    # Consistent color scale across subplots
    vmin = 0.0
    vmax = max(float(f.max()) for f in simulation_results)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    last_im = None
    for ax, idx in zip(axes.flat, idxs):
        frame = simulation_results[idx]
        last_im = ax.imshow(frame[:, :, mid_slice_idx].T, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"t = {idx * dt_sub:.1f}s")
        ax.axis("off")

    fig.suptitle("Concentration evolution (mid-slice)")
    fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.85, label="Concentration")
    plt.show()