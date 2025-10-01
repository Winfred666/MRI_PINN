import numpy as np
from scipy.interpolate import interpn
from scipy.ndimage import zoom as nd_zoom
from typing import Tuple, Literal

InterpMode = Literal["linear", "nearest", "cubic"]

def _symmetrize(t: np.ndarray) -> np.ndarray:
    # (..., 3, 3)
    return 0.5 * (t + np.swapaxes(t, -1, -2))

# Get Frobenius norm of 3x3 matrices in a stack
def _fro_norm(t: np.ndarray) -> np.ndarray:
    # (..., 3, 3) -> (...)
    return np.sqrt(np.sum(t * t, axis=(-2, -1)))

def _logm_spd_stack(t: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    # Vectorized Log for symmetric positive (semi-)definite stack
    # t: (N, 3, 3)
    t = _symmetrize(t)
    w, v = np.linalg.eigh(t)  # (..., 3), (..., 3, 3)
    w_clamped = np.clip(w, eps, None)
    logw = np.log(w_clamped)  # (..., 3)
    # Reconstruct: V * diag(logw) * V^T using broadcasting
    v_scaled = v * logw[..., None, :]
    return v_scaled @ np.swapaxes(v, -1, -2)


def _expm_sym_stack(t: np.ndarray) -> np.ndarray:
    # Vectorized exp for symmetric stack
    t = _symmetrize(t)
    w, v = np.linalg.eigh(t)
    ew = np.exp(w)
    v_scaled = v * ew[..., None, :]
    return v_scaled @ np.swapaxes(v, -1, -2)


def _interpolate_3d(volume: np.ndarray, new_size: Tuple[int, int, int], mode: InterpMode) -> np.ndarray:
    """
    Interpolate a 3D volume to new_size.
    - mode 'linear' or 'nearest' uses interpn with normalized [0,1] grids to match MATLAB interp3 behavior.
    - mode 'cubic' falls back to ndimage.zoom(order=3).
    """
    n1, n2, n3 = volume.shape
    new_n1, new_n2, new_n3 = new_size

    if mode == "cubic":
        # Spline order=3; align_corners-like behavior is not exact but practical
        return nd_zoom(volume, zoom=(new_n1 / max(n1, 1), new_n2 / max(n2, 1), new_n3 / max(n3, 1)),
                       order=3, mode="nearest", grid_mode=True)

    # Build normalized grids [0,1] (handle singleton dims)
    x0 = np.linspace(0.0, 1.0, num=max(n1, 1))
    y0 = np.linspace(0.0, 1.0, num=max(n2, 1))
    z0 = np.linspace(0.0, 1.0, num=max(n3, 1))

    x1 = np.linspace(0.0, 1.0, num=max(new_n1, 1))
    y1 = np.linspace(0.0, 1.0, num=max(new_n2, 1))
    z1 = np.linspace(0.0, 1.0, num=max(new_n3, 1))

    X, Y, Z = np.meshgrid(x1, y1, z1, indexing="ij")
    pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)

    vol_new = interpn(points=(x0, y0, z0), values=volume, xi=pts, method=mode,
                      bounds_error=False, fill_value=0.0)
    return vol_new.reshape(new_n1, new_n2, new_n3)


def resize_dti_log_euclidean(
    D: np.ndarray,
    new_size: Tuple[int, int, int],
    interp: InterpMode = "linear",
    eps: float = 1e-12,
    out_dtype=np.float32,
) -> np.ndarray:
    """
    Resize a DTI tensor field (x, y, z, 3, 3) using Log–Euclidean interpolation.

    Args:
        D: Input tensor field, shape (nx, ny, nz, 3, 3), real-valued.
        new_size: (new_nx, new_ny, new_nz).
        interp: 'linear' (default), 'nearest', or 'cubic' (spline, via ndimage.zoom).
        eps: Eigenvalue clamp for stability before log.
        out_dtype: Output dtype.

    Returns:
        Resized tensor field of shape (new_nx, new_ny, new_nz, 3, 3).
    """
    assert D.ndim == 5 and D.shape[-2:] == (3, 3), "D must be (nx, ny, nz, 3, 3)"
    nx, ny, nz, _, _ = D.shape
    new_nx, new_ny, new_nz = new_size

    D = np.asarray(D)
    in_dtype = D.dtype
    D = D.astype(np.float64, copy=False)

    # Symmetrize
    D = _symmetrize(D)

    # Mask bad tensors
    fro = _fro_norm(D)
    bad = ~np.isfinite(fro) | (fro < 1e-12)
    if bad.any():
        D = D.copy()
        D[bad] = 0.0

    # Log map (vectorized)
    flat = D.reshape(-1, 3, 3)
    log_flat = _logm_spd_stack(flat, eps=eps)
    log_field = log_flat.reshape(nx, ny, nz, 3, 3)

    # Interpolate 6 unique components in log-space
    resized_log = np.zeros((new_nx, new_ny, new_nz, 3, 3), dtype=np.float64)

    for i in range(3):
        for j in range(i, 3):
            vol = log_field[:, :, :, i, j]
            vol_new = _interpolate_3d(vol, (new_nx, new_ny, new_nz), mode=interp)
            vol_new = np.nan_to_num(vol_new, nan=0.0, posinf=0.0, neginf=0.0)
            resized_log[:, :, :, i, j] = vol_new
            resized_log[:, :, :, j, i] = vol_new  # enforce symmetry

    # Exp map back
    resized_log = _symmetrize(resized_log)
    flat_log = resized_log.reshape(-1, 3, 3)
    exp_flat = _expm_sym_stack(flat_log)
    resized = exp_flat.reshape(new_nx, new_ny, new_nz, 3, 3)

    # Re-apply bad mask if resampling to same shape; otherwise leave as-is
    resized = _symmetrize(resized)
    return resized.astype(out_dtype, copy=False)


# Optional quick self-test
if __name__ == "__main__":
    # Create a small random SPD field and resize it
    rng = np.random.default_rng(0)
    nx, ny, nz = 8, 6, 4
    A = rng.standard_normal((nx, ny, nz, 3, 3))
    A = _symmetrize(A)
    # Make SPD by adding multiple of I
    A = A + 3.0 * np.eye(3)[None, None, None, :, :]
    out = resize_dti_log_euclidean(A, (4, 3, 2), interp="linear")
    print("Input:", A.shape, "Output:", out.shape, out.dtype)
    print("Input tensor at (1,1,1):\n", A[1, 1, 1])
    print("Output tensor at (1,1,1):\n", out[1, 1, 1])


def compute_DTI_MD(D: np.ndarray) -> np.ndarray:
    """
    Compute the Mean Diffusivity (MD) from a DTI tensor field.
    Args:
        D: Input tensor field, shape (nx, ny, nz, 3, 3), real-valued.
    Returns:
        Mean diffusivity map, shape (nx, ny, nz).
    """
    assert D.ndim == 5 and D.shape[-2:] == (3, 3), "D must be (nx, ny, nz, 3, 3)"
    # Compute the mean diffusivity by just average trace
    MD = np.trace(D, axis1=-2, axis2=-1) / 3
    return MD

