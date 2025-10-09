import numpy as np
from scipy.ndimage import map_coordinates

def advection_step(C, vx, vy, vz, dt):
    """
    Performs one step of advection using a first-order Semi-Lagrangian method.
    This method is unconditionally stable.

    Args:
        C (np.ndarray): 3D concentration field at time t, shape (nx, ny, nz).
        vx, vy, vz (np.ndarray): 3D velocity component fields (assumed constant for dt).
        dt (float): Time step duration.

    Returns:
        np.ndarray: Advected 3D concentration field at time t + dt.
    """
    nx, ny, nz = C.shape
    
    # 1. Create a grid of destination coordinates (where we want to know the new C)
    x_coords, y_coords, z_coords = np.indices((nx, ny, nz))

    # 2. Calculate the departure points (where the fluid came from)
    # This is the "back-tracing" step. We are looking backward in time.
    # Note: Velocity is assumed to be in voxel units per dt.
    
    departure_x = x_coords - vx * dt
    departure_y = y_coords - vy * dt
    departure_z = z_coords - vz * dt

    # 3. Interpolate the original concentration C at the departure points
    # The map_coordinates function is perfect for this. It handles all the
    # sub-pixel interpolation for us.
    # 'order=1' means linear interpolation, which is fast and usually sufficient.
    # 'mode='nearest'' handles what to do for particles that trace back from outside the domain.
    C_advected = map_coordinates(
        input=C,
        coordinates=[departure_x, departure_y, departure_z],
        order=1,
        mode='nearest' 
    )

    return C_advected

def diffusion_step(C, D, dt, dx, dy, dz, num_sub_steps=5):
    """
    Performs one step of diffusion using an explicit finite difference method.
    This method can handle a constant isotropic (scalar) or a spatially 
    varying full anisotropic (tensor) diffusion coefficient.

    Args:
        C (np.ndarray): 3D concentration field, shape (nx, ny, nz).
        D (float or np.ndarray): Diffusion coefficient. Can be:
            - A single float for constant, isotropic diffusion.
            - An array of shape (nx, ny, nz, 3, 3) for a full diffusion tensor field.
        dt (float): The total time step duration for this diffusion step (e.g., in s).
        dx, dy, dz (float): Voxel dimensions (e.g., in mm).
        num_sub_steps (int): Number of smaller time steps to take for stability.

    Returns:
        np.ndarray: Diffused 3D concentration field.
    """
    sub_dt = dt / num_sub_steps
    C_new = C.copy()

    for _ in range(num_sub_steps):
        # Use np.pad for Neumann boundary conditions (gradient is zero at the boundary)
        C_padded = np.pad(C_new, 1, mode='edge')

        if isinstance(D, (float, int)):
            # Case 1: D is a constant scalar. Use simple Laplacian.
            d2C_dx2 = (C_padded[2:, 1:-1, 1:-1] - 2*C_new + C_padded[:-2, 1:-1, 1:-1]) / (dx*dx)
            d2C_dy2 = (C_padded[1:-1, 2:, 1:-1] - 2*C_new + C_padded[1:-1, :-2, 1:-1]) / (dy*dy)
            d2C_dz2 = (C_padded[1:-1, 1:-1, 2:] - 2*C_new + C_padded[1:-1, 1:-1, :-2]) / (dz*dz)
            C_new += sub_dt * D * (d2C_dx2 + d2C_dy2 + d2C_dz2)
        
        elif D.ndim == 5 and D.shape[-2:] == (3, 3):
            # Case 2: D is a full tensor field (nx, ny, nz, 3, 3).
            # We need to compute ∇ ⋅ (D ∇C).
            
            # 1. Compute gradients of C (dC/dx, dC/dy, dC/dz) at cell centers
            grad_C_x = (C_padded[2:, 1:-1, 1:-1] - C_padded[:-2, 1:-1, 1:-1]) / (2 * dx)
            grad_C_y = (C_padded[1:-1, 2:, 1:-1] - C_padded[1:-1, :-2, 1:-1]) / (2 * dy)
            grad_C_z = (C_padded[1:-1, 1:-1, 2:] - C_padded[1:-1, 1:-1, :-2]) / (2 * dz)

            # 2. Compute the flux vector J = D * ∇C at each cell center
            # Jx = Dxx*dC/dx + Dxy*dC/dy + Dxz*dC/dz
            # Jy = Dyx*dC/dx + Dyy*dC/dy + Dyz*dC/dz
            # Jz = Dzx*dC/dx + Dzy*dC/dy + Dzz*dC/dz
            flux_x = D[..., 0, 0] * grad_C_x + D[..., 0, 1] * grad_C_y + D[..., 0, 2] * grad_C_z
            flux_y = D[..., 1, 0] * grad_C_x + D[..., 1, 1] * grad_C_y + D[..., 1, 2] * grad_C_z
            flux_z = D[..., 2, 0] * grad_C_x + D[..., 2, 1] * grad_C_y + D[..., 2, 2] * grad_C_z

            # Pad the flux components to compute their divergence
            flux_x_padded = np.pad(flux_x, 1, mode='edge')
            flux_y_padded = np.pad(flux_y, 1, mode='edge')
            flux_z_padded = np.pad(flux_z, 1, mode='edge')

            # 3. Compute the divergence of the flux: ∇·J = d(Jx)/dx + d(Jy)/dy + d(Jz)/dz
            div_flux_x = (flux_x_padded[2:, 1:-1, 1:-1] - flux_x_padded[:-2, 1:-1, 1:-1]) / (2 * dx)
            div_flux_y = (flux_y_padded[1:-1, 2:, 1:-1] - flux_y_padded[1:-1, :-2, 1:-1]) / (2 * dy)
            div_flux_z = (flux_z_padded[1:-1, 1:-1, 2:] - flux_z_padded[1:-1, 1:-1, :-2]) / (2 * dz)
            
            divergence_of_flux = div_flux_x + div_flux_y + div_flux_z
            
            C_new += sub_dt * divergence_of_flux
        else:
            raise ValueError(f"Unsupported shape for diffusion coefficient D: {D.shape}")
            
    return C_new

def advect_diffuse_forward_simulation(
    C_initial, vx, vy, vz, D, 
    total_time, num_steps=10, 
    voxel_dims=(0.1, 0.1, 0.1)
):
    """
    Runs a full forward simulation using operator splitting.

    Args:
        C_initial (np.ndarray): The initial 3D concentration field, shape (nx,nz,ny).
        vx, vy, vz (np.ndarray): The static 3D velocity fields, each shape (nx,nz,ny).
        D (float or (x,y,z,3,3) np.ndarray): Diffusion coefficient or tensor (e.g., min^2/min)
        total_time (float): Total simulation time (e.g., 10 min).
        num_steps (int): Number of sub-steps for the simulation, set bigger for more accuracy.
        voxel_dims (tuple): Physical size of a voxel (dx, dy, dz) in mm.

    Returns:
        list[np.ndarray]: A list of the concentration fields at each time step, each shape (nx,ny,nz).
    """
    dt = total_time / num_steps
    dx, dy, dz = voxel_dims

    # --- Important: Convert physical velocity to voxel velocity ---
    # The advection step needs velocity in units of [voxels / dt]
    vx_vox_dt = vx / dx * dt
    vy_vox_dt = vy / dy * dt
    vz_vox_dt = vz / dz * dt

    C_current = C_initial.copy()
    simulation_frames = [C_current]

    print(f"Running simulation {total_time} min config in {num_steps} steps (dt={dt:.2f}min)")
    for i in range(num_steps):
        # Operator Splitting: First advect, then diffuse the result.
        
        # 1. Advection step
        C_advected = advection_step(C_current, vx_vox_dt, vy_vox_dt, vz_vox_dt, dt=1.0) # dt is baked into velocity
        
        # 2. Diffusion step
        C_diffused = diffusion_step(C_advected, D, dt, dx, dy, dz)
        
        C_current = C_diffused
        simulation_frames.append(C_current)
        # if (i+1) % 10 == 0:
        #     print(f"  Step {i+1}/{num_steps} complete.")
    print("Simulation complete.")
    return simulation_frames



# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    # Setup your parameters
    D_coeff = 2.4e-7  # mm^2/s (from the paper)
    voxel_size_mm = 0.1 # mm
    dt_mri = 60 # seconds between MRI frames
    # Assume you have these from your PINN and data:
    # C_observed_t0: The 3D concentration frame at time t=0 (your starting point)
    # velocity_field_u, velocity_field_v, velocity_field_w: The 3D velocity fields (in mm/s)
    # C_observed_t1: The "ground truth" concentration at the next MRI frame (t=60s)

    # --- Generate the data for the example ---
    nx, ny, nz = (64, 64, 32)
    C_observed_t0 = np.zeros((nx, ny, nz))
    half_size = nx // 4  # 32 wide/high total for nx=ny=64
    cx, cy = nx // 2, ny // 2
    x0, x1 = cx - half_size, cx + half_size
    y0, y1 = cy - half_size, cy + half_size
    z0, z1 = nz // 2 - 2, nz // 2 + 2  # keep 4-slice thickness
    C_observed_t0[x0:x1, y0:y1, z0:z1] = 100.0

    # Swirl velocity field around z-axis (constant tangential speed v0)
    v0 = 0.03  # mm/s
    velocity_field_u = np.zeros((nx, ny, nz), dtype=float)
    velocity_field_v = np.zeros((nx, ny, nz), dtype=float)
    velocity_field_w = np.zeros((nx, ny, nz), dtype=float)

    # Center of rotation in physical units (mm)
    x0 = (nx - 1) * voxel_size_mm / 2.0
    y0 = (ny - 1) * voxel_size_mm / 2.0

    for ix in range(nx):
        x_mm = ix * voxel_size_mm
        for iy in range(ny):
            y_mm = iy * voxel_size_mm
            dx_mm = x_mm - x0
            dy_mm = y_mm - y0
            r = np.hypot(dx_mm, dy_mm)

            if r > 1e-12:
                ux = -v0 * (r**2) * (dy_mm / r)  # -sin(theta), bigger r -> faster
                vy =  v0 * (r**2) * (dx_mm / r)  #  cos(theta)
            else:
                ux = 0.0
                vy = 0.0
            for iz in range(nz):
                velocity_field_u[ix, iy, iz] = ux
                velocity_field_v[ix, iy, iz] = vy
                velocity_field_w[ix, iy, iz] = 0.0
    # --- End of example data generation ---


    # Run the simulation from one MRI frame to the next
    # Use several sub-steps for accuracy
    simulation_results = advect_diffuse_forward_simulation(
        C_initial=C_observed_t0,
        vx=velocity_field_u,
        vy=velocity_field_v,
        vz=velocity_field_w,
        D=D_coeff,
        total_time=dt_mri,
        num_steps=50, # More sub-steps lead to higher accuracy
        voxel_dims=(voxel_size_mm, voxel_size_mm, voxel_size_mm)
    )

    # The final frame of the simulation is your prediction for the next MRI frame
    C_predicted_t1 = simulation_results[-1]

    # You can also visualize the results (e.g., using matplotlib)
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
        last_im = ax.imshow(frame[:, :, mid_slice_idx], vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_title(f"t = {idx * dt_sub:.1f}s")
        ax.axis("off")

    fig.suptitle("Concentration evolution (mid-slice)")
    fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.85, label="Concentration")
    plt.show()