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



def save_velocity_mat(vx,vy,vz, pixdim, D, use_DTI ,path="data/ad_net_velocity.mat"):
    v_tensor = np.stack([vx.flatten(), vy.flatten(), vz.flatten()], axis=1)  # shape (N, 3)
    v_tensor /= pixdim  # convert to cell/min
    v_tensor = v_tensor.reshape(-1, 1)  # shape (N*3, 1)
    # repeated 4 times, as in rOMT there are 4 interpolate steps
    v_tensor = np.tile(v_tensor, (4, 1))
    print(v_tensor.shape, ' velocity save to: ', path)
    sio.savemat(path, {"u": v_tensor,
                      "D": D,
                      "use_DTI":use_DTI})

from utils.process_DTI import resize_dti_log_euclidean, compute_DTI_MD


def load_DTI(char_domain, path="data/DCE_nii_data/dti_tensor_3_3.mat", resize_to=(32, 24, 16)):
    DTI_tensor = sio.loadmat(path)["D_tensor"]
    print("Original DTI shape: ", DTI_tensor.shape) # (X,Y,Z,3,3)
    # need to resize it to the same shape as data
    DTI_tensor = resize_dti_log_euclidean(DTI_tensor, resize_to)
    print("Resized DTI shape: ", DTI_tensor.shape)

    # WARNING: in the experiments, the tracer's DTI is about 1/3 of water's DTI
    DTI_tensor = DTI_tensor * 60.0 / (char_domain.L_star**2/char_domain.T_star * 3)

    # sanity check MD
    DTI_MD = compute_DTI_MD(DTI_tensor)
    print("DTI_MD min: ", DTI_MD.min(), "DTI_MD max: ", DTI_MD.max(), "DTI_MD mean: ", DTI_MD.mean())

    return DTI_tensor,DTI_MD
