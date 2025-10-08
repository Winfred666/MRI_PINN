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
    front_values = [30, 40] # Example concentration levels

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
                
                # Get the normal vector at P1 (use gradients in x,y,z order)
                normal = np.array([grad_x[ix, iy, iz], grad_y[ix, iy, iz], grad_z[ix, iy, iz]])
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
                search_radius = 4.0 # In voxels, adjust as needed

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