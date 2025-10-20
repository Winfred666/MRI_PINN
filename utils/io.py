import scipy.io as sio
import numpy as np

def load_dcemri_data(path):
    # 1. load training data (DCE-MRI)
    downsample_data = np.load(path)
    data = downsample_data["data"]
    mask = downsample_data["mask"]
    pixdim = downsample_data["pixdim"]
    x, y, z, t = (
        downsample_data["x"],
        downsample_data["y"],
        downsample_data["z"],
        downsample_data["t"],
    )
    print("data_shape: ", data.shape, "pixdim: ", pixdim)
    print("domain_shape: ",x.shape, y.shape, z.shape, t.shape)  # (nx,), (ny,), (nz,), (nt,)
    print("min_c: ", data.min(), "max_c: ", data.max())
    return data, mask, pixdim, x, y, z, t



def save_velocity_mat(vx, vy, vz, pixdim, D, use_DTI, path="data/ad_net_velocity_corrected.mat"):
    # vx, vy, vz: each shape (nx, ny, nz)
    # Total number of voxels
    N = vx.size 
    # 1. Flatten each component separately, and pply the scaling factor
    # This creates contiguous blocks of x, y, and z data
    vx_flat = vx.flatten() / pixdim[0] # convert to cell/min, shaped (N,)
    vy_flat = vy.flatten() / pixdim[1]
    vz_flat = vz.flatten() / pixdim[2]

    # 2. Concatenate these blocks to form the vector for a single time step.
    # The order is [all_x, all_y, all_z], which is what MATLAB's reshape expects.
    v_single_timestep = np.concatenate([vx_flat, vy_flat, vz_flat]) # Shape (N*3,)

    # 3. Repeat this block 4 times for the time dimension (nt=4)
    # This creates [ts1_x, ts1_y, ts1_z, ts2_x, ts2_y, ts2_z, ...]
    v_tensor = np.tile(v_single_timestep, 4) # Shape (N*3*4,)

    # 4. Reshape to a column vector (N*3*4, 1) for saving, which is good practice.
    v_tensor = v_tensor.reshape(-1, 1)

    print(f"Corrected velocity shape: {v_tensor.shape}, saved to: {path}")
    sio.savemat(path, {"u": v_tensor,
                       "D": D,
                       "use_DTI": use_DTI})



from utils.process_DTI import resize_dti_log_euclidean, compute_DTI_MD

def load_DTI(char_domain, path="data/DCE_nii_data/dti_tensor_3_3.mat", resize_to=(32, 24, 16)):
    DTI_tensor_raw = sio.loadmat(path)["D_tensor"]
    print("Original DTI shape: ", DTI_tensor_raw.shape) # (X,Y,Z,3,3)
    # need to resize it to the same shape as data
    DTI_tensor_raw = resize_dti_log_euclidean(DTI_tensor_raw, resize_to)
    print("Resized DTI shape: ", DTI_tensor_raw.shape)
    
    # transform from mm^2/min (length mm is not affected by resize) to char_domain units
    
    # As DTI is (x,y,z,3,3), and L_star is (3,), stretch accordingly
    # D_char = D_phys / (L_star_i * L_star_j)
    # This can be done via broadcasting:
    L_star_col = char_domain.L_star.reshape(1, 1, 1, 3, 1)
    L_star_row = char_domain.L_star.reshape(1, 1, 1, 1, 3)
    DTI_tensor = DTI_tensor_raw / (L_star_col * L_star_row)

    # WARNING: in the experiments, the tracer's DTI is about 1/3 of water's DTI, will be add later in Pe_g.
    DTI_tensor = DTI_tensor / (1.0/char_domain.T_star) # D has units L^2/T, so we divide by L^2/T_char

    # sanity check MD
    DTI_MD = compute_DTI_MD(DTI_tensor_raw)
    print("DTI_MD min: ", DTI_MD.min(), "DTI_MD max: ", DTI_MD.max(), "DTI_MD mean: ", DTI_MD.mean())

    return DTI_tensor, DTI_tensor_raw,DTI_MD
