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

    # STEP 1: Choose front values
    front_values = [30, 40, 50] # Example concentration levels

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
                
                # Get the normal vector at P1 (use gradients in x,y,z order)
                normal = np.array([grad_x[ix, iy, iz], grad_y[ix, iy, iz], grad_z[ix, iy, iz]])
                norm_mag = np.linalg.norm(normal)
                
                if norm_mag < 1e-6: # Avoid division by zero
                    continue
                
                unit_normal = normal / norm_mag
                
                # STEP 4: A simplified search for the corresponding point P2
                # (More advanced methods exist, but nearest neighbor along normal is a good start)
                # Find the closest point on Front_2 to P1
                dist, idx = tree_front2.query(p1)
                p2 = verts2[idx]
                
                # Project displacement onto the normal direction
                displacement_vec = p2 - p1
                displacement_mag = np.dot(displacement_vec, unit_normal)
                
                # STEP 5: Calculate velocity
                velocity_mag = displacement_mag / dt
                velocity_vec = velocity_mag * unit_normal
                
                # Store the result in (x, y, z, u, v, w)
                sparse_velocities.append([p1[0], p1[1], p1[2], velocity_vec[0], velocity_vec[1], velocity_vec[2]])

    # STEP 7: Interpolate sparse data to a dense grid
    sparse_velocities = np.array(sparse_velocities)
    if sparse_velocities.size == 0:
        # No fronts found; return zero field
        return np.zeros((nx, ny, nz, 3), dtype=float)

    points = sparse_velocities[:, :3]
    values = sparse_velocities[:, 3:]
    
    # Create the grid to interpolate onto (x, y, z)
    grid_x, grid_y, grid_z = np.mgrid[0:nx, 0:ny, 0:nz]
    
    initial_velocity_field = griddata(points, values, (grid_x, grid_y, grid_z), method='linear', fill_value=0.0)
    
    return initial_velocity_field # shape: (nx, ny, nz, 3)