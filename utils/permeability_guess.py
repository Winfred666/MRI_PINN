import numpy as np
import scipy.ndimage as ndi

def estimate_initial_permeability(ser_data, time_points, 
                                  ser_threshold=150.0, 
                                  time_threshold_min=16.0,
                                  high_perm_value=1e-6,
                                  low_perm_value=1e-10):
    """
    Estimates the initial permeability map based on tracer arrival time.

    Args:
        ser_data (np.ndarray): 4D array (time, z, y, x) of SER values.
        time_points (np.ndarray): 1D array of time points in minutes.
        ser_threshold (float): SER value to define tracer arrival.
        time_threshold_min (float): Time in minutes to define "quick" arrival.
        high_perm_value (float): Value for high permeability regions.
        low_perm_value (float): Value for low permeability regions.

    Returns:
        np.ndarray: 3D array (z, y, x) of the initial permeability guess.
    """
    
    # --- Step 1: Create a boolean mask of where the tracer arrived ---
    # This mask will be True for any voxel that EVER crosses the SER threshold.
    # The shape will be (nz, ny, nx).
    arrival_mask = np.any(ser_data > ser_threshold, axis=0)
    
    # --- Step 2: Calculate the "Arrival Time Map" ---
    # We want to find the time of FIRST passage across the threshold for each voxel.
    
    # Create a 4D boolean array where SER > threshold.
    exceeds_threshold_4d = ser_data > ser_threshold
    
    # The 'argmax' trick: For each voxel, argmax along the time axis will return
    # the INDEX of the first True value. This is a very efficient way to find
    # the first time of passage.
    # Note: If a voxel never exceeds the threshold, argmax returns 0.
    arrival_time_indices = np.argmax(exceeds_threshold_4d, axis=0)
    
    # Convert these indices to actual time in minutes
    arrival_time_map = time_points[arrival_time_indices]
    
    # IMPORTANT Correction for the argmax trick:
    # Voxels that never arrive will have their time incorrectly set to time_points[0] (e.g., 0 minutes).
    # We must correct this. We use the arrival_mask from Step 1.
    # Where the tracer never arrived, set the arrival time to infinity.
    arrival_time_map[~arrival_mask] = np.inf
    
    # --- Step 3: Apply the two conditions to create the high permeability mask ---
    # Condition 1 is implicitly handled by the np.inf above.
    # Condition 2: Arrival time must be less than the time threshold.
    is_fast_arrival = arrival_time_map < time_threshold_min
    
    # The high permeability mask is where the arrival was fast.
    high_perm_mask = is_fast_arrival # Shape: (nz, ny, nx)
    
    # --- Step 4: Create the final permeability map ---
    # Initialize the entire map with the low permeability value.
    initial_permeability_map = np.full(ser_data.shape[1:], low_perm_value, dtype=np.float32)
    
    # Use the mask to set the high permeability values.
    initial_permeability_map[high_perm_mask] = high_perm_value
    
    # --- Optional Step 5: Smoothing (as mentioned in the paper) ---
    # The paper mentions they smooth this binary map with a Gaussian filter
    # to create more realistic, less sharp transitions.
    # "We smoothed the binary permeability map using a 3D Gaussian filter for discrete instability..."
    
    # sigma=1.0 means a standard deviation of 1 voxel.
    smoothed_perm_map = ndi.gaussian_filter(initial_permeability_map, sigma=1.0)

    # Rescaling after smoothing to restore the max value.
    # Smoothing can lower the peak value, so this step ensures the "highway"
    # permeability is still at its intended maximum.
    if smoothed_perm_map.max() > 0:
        smoothed_perm_map = (smoothed_perm_map / smoothed_perm_map.max()) * high_perm_value
    
    # Return both the sharp binary map and a smoothed version
    return initial_permeability_map, smoothed_perm_map
