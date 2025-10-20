# use front-tracking technique to estimate velocity field

import numpy as np
import scipy.ndimage as ndi
from scipy.interpolate import griddata
from scipy.spatial import KDTree
from skimage.measure import marching_cubes

# Assume 'concentration_data' is a 4D NumPy array (x, y, z, t)
# Assume 'dt' is the time between frames (e.g., 60 seconds)
def front_tracking_velocity(concentration_data, dt):
    # Adapted to x, y, z, t ordering
    nx, ny, nz, num_times = concentration_data.shape
    # Store the sparse velocity data: [x, y, z, u, v, w]
    sparse_velocities = []

    # STEP 1: Choose front values(already scale to )
    front_values = [20, 30, 40] # Example concentration levels

    # Loop over time and fronts
    for t_idx in range(num_times - 1):
        # Take 3D volumes at consecutive times (x, y, z)
        t1_data = concentration_data[..., t_idx]
        t2_data = concentration_data[..., t_idx + 1]

        # STEP 3: Pre-calculate the gradient field for t1
        # Use a Gaussian filter to smooth the data first, as gradients are noise-sensitive
        smoothed_t1_data = ndi.gaussian_filter(t1_data, sigma=1.0)
        # Gradients now follow (x, y, z) axis order
        grad_x, grad_y, grad_z = np.gradient(smoothed_t1_data)
        


        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
        max_grad = np.max(gradient_magnitude)
        if max_grad > 1e-8:
            soft_mask = gradient_magnitude / max_grad
        else:
            soft_mask = np.zeros_like(gradient_magnitude)

        for c_front in front_values:
            try:
                # STEP 2: Extract the 3D front geometry using Marching Cubes
                # With (x, y, z) volume, returned vertices are (x, y, z) index coordinates
                verts1, _, _, _ = marching_cubes(t1_data, level=c_front)
                verts2, _, _, _ = marching_cubes(t2_data, level=c_front)

                if len(verts1) == 0 or len(verts2) == 0:
                    continue
                
                # For efficient searching on the second front
                tree_front2 = KDTree(verts2)

            except (ValueError, RuntimeError): # Marching cubes can fail if front is not found
                continue

            # STEP 3-5: Calculate velocity for points on the first front
            for p1 in verts1:
                # Get integer indices for gradient lookup (x, y, z)
                ix, iy, iz = map(int, p1)
                
                # Get the normal vector at P1, pointed outward (use gradients in x,y,z order)
                normal = -np.array([grad_x[ix, iy, iz], grad_y[ix, iy, iz], grad_z[ix, iy, iz]])
                norm_mag = np.linalg.norm(normal)
                
                if norm_mag < 1e-6: # Avoid division by zero
                    continue
                
                unit_normal = normal / norm_mag
                # A simplified version of STEP 4: Find closest point on front2, not following normal direction.
                # dist, idx = tree_front2.query(p1)
                # p2 = verts2[idx]

                # STEP 4: A more robust search for the corresponding point P2
                # Define a maximum search distance, similar to dmax in the paper.
                # This should be larger than the expected movement in one time step.
                search_radius = 5.0 # In voxels, adjust as needed

                # Find all candidate points on front_2 within the search radius of p1
                indices_in_radius = tree_front2.query_ball_point(p1, r=search_radius)

                if not indices_in_radius:
                    # No point found nearby, cannot calculate velocity here
                    continue

                candidate_points = verts2[indices_in_radius]

                # Calculate displacement vectors from p1 to all candidate points
                displacement_vectors = candidate_points - p1

                # Find the displacement vector that is most aligned with the normal vector.
                # We do this by finding the maximum dot product.
                # We don't need to normalize the vectors, as argmax will find the same index.
                dot_products = np.dot(displacement_vectors, unit_normal)
                best_candidate_local_idx = np.argmax(dot_products)

                # if best corresponding point p2 still negative, fail to find initial velocity
                if dot_products[best_candidate_local_idx] < 0.5:
                    continue
                p2 = candidate_points[best_candidate_local_idx]

                # Now, we use the *actual* displacement vector to this specific p2
                displacement_vec = p2 - p1
                # And project its magnitude onto the normal direction for the speed
                displacement_mag = np.dot(displacement_vec, unit_normal)
                
                # STEP 5: Calculate velocity
                velocity_mag = displacement_mag / dt
                velocity_vec = velocity_mag * unit_normal
                
                # STEP 6: Apply a soft mask based on gradient magnitude to reduce noise impact
                velocity_vec *= soft_mask[ix, iy, iz]
                # Store the result in (x, y, z, u, v, w)
                sparse_velocities.append([p1[0], p1[1], p1[2], velocity_vec[0], velocity_vec[1], velocity_vec[2]])
        

    # STEP 8: Interpolate sparse data to a dense grid
    sparse_velocities = np.array(sparse_velocities)
    if sparse_velocities.size == 0:
        # No fronts found; return zero field
        return np.zeros((nx, ny, nz, 3), dtype=float)

    points = sparse_velocities[:, :3]
    values = sparse_velocities[:, 3:]
    
    # Create the grid to interpolate onto (x, y, z)
    grid_x, grid_y, grid_z = np.mgrid[0:nx, 0:ny, 0:nz]
    
    initial_velocity_field = griddata(points, values, 
                                      (grid_x, grid_y, grid_z), method='linear', fill_value=0.0)
    
    
    return initial_velocity_field # shape: (nx, ny, nz, 3)





if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # --- 1. Create a simple 3D example: an expanding sphere ---
    
    # Define grid parameters
    grid_size = 50
    num_times = 5
    dt = 1.0  # Time step duration

    # Create a 4D array to hold the concentration data (x, y, z, t)
    concentration_data = np.zeros((grid_size, grid_size, grid_size, num_times))

    # Define the center of the grid
    center = np.array([grid_size / 2, grid_size / 2, grid_size / 2])

    # Create coordinate grids
    x, y, z = np.mgrid[0:grid_size, 0:grid_size, 0:grid_size]

    # Generate the expanding sphere over time
    print("Generating synthetic data of an expanding sphere...")
    initial_radius = 8
    expansion_speed = 2.0 # voxels per time step
    
    for t_idx in range(num_times):
        current_radius = initial_radius + t_idx * expansion_speed
        # Calculate distance of each voxel from the center
        distance_from_center = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
        
        # Use a smooth transition (sigmoid) to define the sphere's boundary
        # This helps marching_cubes and gradient calculations
        # The concentration goes from ~100 inside to ~0 outside, centered at `current_radius`
        concentration_data[..., t_idx] = 50 * (1 - np.tanh((distance_from_center - current_radius) / 2.0))

    print("Synthetic data generated.")

    # --- 2. Run the front_tracking_velocity function ---
    print("Estimating velocity field using front_tracking_velocity...")
    estimated_velocity = front_tracking_velocity(concentration_data, dt)
    print("Velocity field estimated.")

    # --- 3. Visualize the results ---
    
    # Create a figure with three subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # --- Plot 1: Concentration at the first time step ---
    central_slice_idx = grid_size // 2
    ax = axes[0]
    im = ax.imshow(concentration_data[:, :, central_slice_idx, 0].T, 
                   cmap='viridis', origin='lower', extent=[0, grid_size, 0, grid_size])
    ax.set_title(f'Concentration at t=0 (Z-slice={central_slice_idx})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Plot 2: Concentration at the last time step ---
    ax = axes[1]
    im = ax.imshow(concentration_data[:, :, central_slice_idx, -1].T, 
                   cmap='viridis', origin='lower', extent=[0, grid_size, 0, grid_size])
    ax.set_title(f'Concentration at t={num_times-1} (Z-slice={central_slice_idx})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # --- Plot 3: Estimated velocity field (quiver plot) ---
    ax = axes[2]
    
    # Extract velocity components for the central slice
    u = estimated_velocity[:, :, central_slice_idx, 0]
    v = estimated_velocity[:, :, central_slice_idx, 1]
    
    # Subsample the grid for a cleaner plot
    subsample = 4
    x_sub = x[::subsample, ::subsample, central_slice_idx]
    y_sub = y[::subsample, ::subsample, central_slice_idx]
    u_sub = u[::subsample, ::subsample]
    v_sub = v[::subsample, ::subsample]

    # Calculate magnitude for coloring the arrows
    magnitude = np.sqrt(u_sub**2 + v_sub**2)

    # Plot the quiver, with color representing magnitude
    q = ax.quiver(x_sub, y_sub, u_sub, v_sub, magnitude, cmap='jet', scale=25, headwidth=4)
    fig.colorbar(q, ax=ax, fraction=0.046, pad=0.04, label='Velocity Magnitude (voxels/dt)')
    ax.set_title(f'Estimated Velocity Field (Z-slice={central_slice_idx})')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    plt.tight_layout()
    plt.show()
