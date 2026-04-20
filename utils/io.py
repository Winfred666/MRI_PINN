import os
import re
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy.io as sio
import torch


_DCE_FILENAME_RE = re.compile(r"^(?:p)?dce2base_(\d+)\.nii(?:\.gz)?$")
_DCE_TEMPLATE_RE = re.compile(r"\{(?::)?0?(\d+)d\}")


def _unwrap_pt_samples(samples):
    if isinstance(samples, dict):
        return samples
    if isinstance(samples, (list, tuple)):
        if len(samples) == 0:
            raise ValueError("PT payload field 'samples' must not be empty.")
        sample = samples[0]
        if not isinstance(sample, dict):
            raise ValueError(
                "PT payload field 'samples' must contain dict entries when provided as a list/tuple."
            )
        return sample
    raise ValueError("PT payload field 'samples' must be a dict or a non-empty list/tuple of dicts.")


def load_dce_mri_pt(path: str):
    pt_data = torch.load(path, map_location="cpu")
    if not isinstance(pt_data, dict):
        raise ValueError(f"Expected PT payload to be a dict, got {type(pt_data)} from {path}")
    if "samples" not in pt_data:
        raise KeyError(f"PT payload is missing required key 'samples': {path}")
    if "dx" not in pt_data:
        raise KeyError(f"PT payload is missing required key 'dx': {path}")

    sample = _unwrap_pt_samples(pt_data["samples"])
    if "x" not in sample:
        raise KeyError(f"PT payload sample is missing required key 'x': {path}")

    dcemri = sample["x"]
    if isinstance(dcemri, torch.Tensor):
        dcemri = dcemri.detach().cpu().numpy()
    dcemri = np.asarray(dcemri, dtype=np.float32)
    if dcemri.ndim != 4:
        raise ValueError(f"Expected PT DCE sample 'x' to have shape (T, D, H, W), got {dcemri.shape}")

    nt, nx, ny, nz = dcemri.shape
    data = np.transpose(dcemri, (1, 2, 3, 0))  # (nx, ny, nz, nt)

    pixdim = np.asarray(pt_data["dx"], dtype=np.float32).reshape(-1)
    if pixdim.size == 1:
        pixdim = np.repeat(pixdim[0], 3)
    if pixdim.size != 3:
        raise ValueError(f"Expected PT payload 'dx' to contain 3 voxel spacings, got {tuple(pixdim)}")

    mask = np.any(data != 0, axis=-1)
    if not np.any(mask):
        mask = np.ones(data.shape[:3], dtype=bool)

    x = np.arange(nx, dtype=np.float32) * pixdim[0]
    y = np.arange(ny, dtype=np.float32) * pixdim[1]
    z = np.arange(nz, dtype=np.float32) * pixdim[2]
    t = np.arange(nt, dtype=np.float32) * 1.0

    return data, mask, pixdim, x, y, z, t


def _parse_frame_number(filename: str):
    match = _DCE_FILENAME_RE.match(filename)
    if not match:
        return None
    return int(match.group(1))


def _normalize_dce_template(template_path: str):
    template_str = str(template_path).strip()
    if not template_str:
        raise ValueError("DCE template path must be a non-empty string.")

    placeholder_match = _DCE_TEMPLATE_RE.search(template_str)
    if placeholder_match is None:
        raise ValueError(
            "Template path must contain a frame placeholder like '{:04d}' or '{04d}'. "
            f"Received: {template_path}"
        )

    width = int(placeholder_match.group(1))
    python_template = _DCE_TEMPLATE_RE.sub(f"{{:0{width}d}}", template_str, count=1)
    return python_template, width


def _collect_dce_frame_files(dce_dir: str):
    dir_path = Path(dce_dir)
    if not dir_path.is_dir():
        raise FileNotFoundError(f"DCE directory not found: {dce_dir}")

    frame_to_file = {}
    for file_path in sorted(dir_path.iterdir()):
        if not file_path.is_file():
            continue
        frame_num = _parse_frame_number(file_path.name)
        if frame_num is None:
            continue

        # Prefer .nii.gz over .nii when both exist.
        current = frame_to_file.get(frame_num)
        if current is None:
            frame_to_file[frame_num] = file_path
            continue
        if str(file_path).endswith(".nii.gz"):
            frame_to_file[frame_num] = file_path

    if not frame_to_file:
        raise ValueError(f"No DCE frame files matched 'dce2base_*.nii*' under: {dce_dir}")
    return frame_to_file


def _load_dcemri_nifti_dir(path: str, frame_numbers=None):
    frame_to_file = _collect_dce_frame_files(path)
    available_frames = sorted(frame_to_file.keys())

    if frame_numbers is None:
        selected_frames = available_frames
    else:
        selected_frames = [int(v) for v in frame_numbers]
        if len(selected_frames) == 0:
            raise ValueError("frame_numbers must not be empty when provided.")
        missing = [f for f in selected_frames if f not in frame_to_file]
        if missing:
            raise ValueError(
                f"Requested frame_numbers {missing} are missing in '{path}'. "
                f"Available range: {available_frames[0]}..{available_frames[-1]}"
            )

    selected_files = [frame_to_file[f] for f in selected_frames]
    first_img = nib.load(str(selected_files[0]))
    first_data = np.asarray(first_img.dataobj, dtype=np.float32)
    if first_data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI frames, got shape {first_data.shape} for {selected_files[0]}")

    frame_arrays = [first_data]
    for file_path in selected_files[1:]:
        frame_data = np.asarray(nib.load(str(file_path)).dataobj, dtype=np.float32)
        if frame_data.shape != first_data.shape:
            raise ValueError(
                f"Inconsistent frame shape for {file_path}: {frame_data.shape}, expected {first_data.shape}"
            )
        frame_arrays.append(frame_data)

    data = np.stack(frame_arrays, axis=-1)  # (nx, ny, nz, nt)
    mask = np.ones(first_data.shape, dtype=bool)

    zooms = np.asarray(first_img.header.get_zooms()[:3], dtype=np.float32)
    if zooms.shape[0] != 3:
        raise ValueError(f"Invalid pixdim in NIfTI header for {selected_files[0]}: {zooms}")
    pixdim = zooms

    nx, ny, nz = first_data.shape
    x = np.arange(nx, dtype=np.float32) * pixdim[0]
    y = np.arange(ny, dtype=np.float32) * pixdim[1]
    z = np.arange(nz, dtype=np.float32) * pixdim[2]

    # Keep the conventional 4 min interval used by existing datasets.
    t = np.asarray(selected_frames, dtype=np.float32) * 4.0

    return data, mask, pixdim, x, y, z, t


def _load_dcemri_nifti_template(path: str, frame_numbers=None):
    if frame_numbers is None or len(frame_numbers) == 0:
        raise ValueError("Template-style DCE inputs require explicit frame_numbers.")

    template_path, _ = _normalize_dce_template(path)
    selected_frames = [int(v) for v in frame_numbers]
    selected_files = []
    missing_files = []
    for frame_num in selected_frames:
        file_path = Path(template_path.format(frame_num))
        if not file_path.is_file():
            missing_files.append(str(file_path))
        else:
            selected_files.append(file_path)

    if missing_files:
        raise FileNotFoundError(
            "Missing DCE frame files for template input:\n" + "\n".join(missing_files)
        )

    first_img = nib.load(str(selected_files[0]))
    first_data = np.asarray(first_img.dataobj, dtype=np.float32)
    if first_data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI frames, got shape {first_data.shape} for {selected_files[0]}")

    frame_arrays = [first_data]
    for file_path in selected_files[1:]:
        frame_data = np.asarray(nib.load(str(file_path)).dataobj, dtype=np.float32)
        if frame_data.shape != first_data.shape:
            raise ValueError(
                f"Inconsistent frame shape for {file_path}: {frame_data.shape}, expected {first_data.shape}"
            )
        frame_arrays.append(frame_data)

    data = np.stack(frame_arrays, axis=-1)
    mask = np.ones(first_data.shape, dtype=bool)

    zooms = np.asarray(first_img.header.get_zooms()[:3], dtype=np.float32)
    if zooms.shape[0] != 3:
        raise ValueError(f"Invalid pixdim in NIfTI header for {selected_files[0]}: {zooms}")
    pixdim = zooms

    nx, ny, nz = first_data.shape
    x = np.arange(nx, dtype=np.float32) * pixdim[0]
    y = np.arange(ny, dtype=np.float32) * pixdim[1]
    z = np.arange(nz, dtype=np.float32) * pixdim[2]
    t = np.asarray(selected_frames, dtype=np.float32) * 4.0

    return data, mask, pixdim, x, y, z, t


def _load_brain_mask(mask_path: str, expected_shape):
    mask_img = nib.load(mask_path)
    mask_data = np.asarray(mask_img.dataobj)
    if mask_data.ndim > 3:
        mask_data = np.squeeze(mask_data)
    if mask_data.shape != tuple(expected_shape):
        raise ValueError(
            f"brain_mask shape mismatch: {mask_data.shape} vs expected {tuple(expected_shape)} "
            f"for training data volume."
        )
    return mask_data > 0


def load_dcemri_data(path, brain_mask_path=None, frame_numbers=None):
    path_str = str(path)
    if os.path.isdir(path_str):
        data, mask, pixdim, x, y, z, t = _load_dcemri_nifti_dir(path_str, frame_numbers=frame_numbers)
        source_desc = "nifti_dir"
    elif "{" in path_str and "}" in path_str:
        data, mask, pixdim, x, y, z, t = _load_dcemri_nifti_template(path_str, frame_numbers=frame_numbers)
        source_desc = "nifti_template"
    elif os.path.isfile(path_str):
        suffix = Path(path_str).suffix.lower()
        if suffix == ".pt":
            data, mask, pixdim, x, y, z, t = load_dce_mri_pt(path_str)
            source_desc = "pt"
        else:
            raise ValueError(
                f"Unsupported DCE file format '{suffix}' for path: {path}. "
                "Use .pt samples or NIfTI directory/template inputs."
            )
    else:
        raise FileNotFoundError(f"DCE input path not found: {path}")

    if brain_mask_path is not None:
        mask = _load_brain_mask(str(brain_mask_path), expected_shape=data.shape[:3])
        print(f"Using explicit brain mask: {brain_mask_path}")

    print(f"data_source: {source_desc}")
    print("data_shape: ", data.shape, "pixdim: ", pixdim)
    print("domain_shape: ", x.shape, y.shape, z.shape, t.shape)  # (nx,), (ny,), (nz,), (nt,)
    print("min_c: ", data.min(), "max_c: ", data.max())
    print("mask_voxels: ", int(mask.sum()))
    return data, mask, pixdim, x, y, z, t



def save_velocity_pt(vx, vy, vz, pixdim, D, use_DTI, path="data/ad_net_velocity_physics.pt"):
    """Persist the recovered 3D velocity field directly in physical units."""
    vx = np.asarray(vx, dtype=np.float32)
    vy = np.asarray(vy, dtype=np.float32)
    vz = np.asarray(vz, dtype=np.float32)
    pixdim = np.asarray(pixdim, dtype=np.float32)

    if vx.shape != vy.shape or vx.shape != vz.shape:
        raise ValueError(f"Velocity component shape mismatch: vx {vx.shape}, vy {vy.shape}, vz {vz.shape}")

    velocity_3d_mm_min = np.stack([vx, vy, vz], axis=-1)

    payload = {
        "vx_mm_min": torch.from_numpy(vx.copy()),
        "vy_mm_min": torch.from_numpy(vy.copy()),
        "vz_mm_min": torch.from_numpy(vz.copy()),
        "velocity_3d_mm_min": torch.from_numpy(velocity_3d_mm_min.copy()),
        "pixdim_mm": torch.from_numpy(pixdim.copy()),
        "D": float(D),
        "use_DTI": bool(use_DTI),
        "unit": "mm/min",
        "component_order": ("vx", "vy", "vz"),
        "layout": "(nx, ny, nz, 3)",
    }

    print(f"Saving 3D velocity field to: {path}, shape={velocity_3d_mm_min.shape}")
    torch.save(payload, path)



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
