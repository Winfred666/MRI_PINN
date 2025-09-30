import numpy as np

import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display

# 1. visualize 3D velocity field quiver with interactive widget or window
def draw_3d_quiver(X, Y, Z, vx, vy, vz, stride=3, length_scale=0.25, cmap='bwr', normalize_vectors=False):
    """
    Colored 3D quiver: arrow length + color encode speed magnitude.
    Parameters
      stride: subsampling step
      length_scale: global scaling factor for arrows (after normalization)
      cmap: matplotlib colormap (bwr gives blue->white->red)
      normalize_vectors: if True all arrows same length; color still encodes magnitude
    """
    # Subsample
    Xs = X[::stride, ::stride, ::stride]
    Ys = Y[::stride, ::stride, ::stride]
    Zs = Z[::stride, ::stride, ::stride]
    U  = vx[::stride, ::stride, ::stride]
    V  = vy[::stride, ::stride, ::stride]
    W  = vz[::stride, ::stride, ::stride]

    # Flatten for quiver
    Xf = Xs.ravel(); Yf = Ys.ravel(); Zf = Zs.ravel()
    Uf = U.ravel();  Vf = V.ravel();  Wf = W.ravel()

    mag = np.sqrt(Uf**2 + Vf**2 + Wf**2)
    mag_min, mag_max = mag.min() if mag.size else 0.0, mag.max() if mag.size else 1.0
    mag_rng = (mag_max - mag_min) if mag_max > mag_min else 1.0
    mag_norm = (mag - mag_min) / mag_rng  # 0..1 for colormap

    if normalize_vectors:
        # Keep direction only, scale uniformly
        nonzero = mag > 0
        Uf[nonzero] /= mag[nonzero]
        Vf[nonzero] /= mag[nonzero]
        Wf[nonzero] /= mag[nonzero]
        mag_for_length = np.ones_like(mag)
    else:
        # Scale by magnitude
        max_m = mag.max() if mag.size else 1.0
        if max_m > 0:
            Uf /= max_m
            Vf /= max_m
            Wf /= max_m
        mag_for_length = mag

    # Apply global length scale
    Uf *= length_scale
    Vf *= length_scale
    Wf *= length_scale

    colors = plt.get_cmap(cmap)(mag_norm)

    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Matplotlib 3D quiver does not produce an easy colorbar; create a dummy scatter for colorbar
    sc = ax.scatter(Xf, Yf, Zf, c=mag, cmap=cmap, alpha=0.0)
    q = ax.quiver(Xf, Yf, Zf, Uf, Vf, Wf, colors=colors, length=1.0, normalize=False)

    cbar = fig.colorbar(sc, ax=ax, shrink=0.65, pad=0.1)
    cbar.set_label('|v| (magnitude)')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Velocity Field (colored by |v|)')
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()
    plt.show()
    return fig

def interactive_quiver(vx, vy, vz, pixdim):
    X, Y, Z = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]), np.arange(vx.shape[2]), indexing='ij')
    X = X * pixdim[0]
    Y = Y * pixdim[1]
    Z = Z * pixdim[2]

    def _show(stride=3, length_scale=25, normalize_vectors=False):
        plt.close('all')
        draw_3d_quiver(
            X, Y, Z, vx, vy, vz,
            stride=stride,
            length_scale=length_scale/100.0,
            normalize_vectors=normalize_vectors
        )
    interact(
        _show,
        stride=widgets.IntSlider(min=1, max=10, step=1, value=3, description='Stride'),
        length_scale=widgets.IntSlider(min=1, max=100, step=1, value=25, description='Len %'),
        normalize_vectors=widgets.Checkbox(value=False, description='Uniform length')
    )



# 2. visualize the slices with mask + thresholding
def draw_nifti_slices_with_threshold(img, brain_mask=None):
    # Get array (memory-mapped if possible)
    data = img
    vol = data[..., 0] if data.ndim == 4 else data
    print("Volume shape:", vol.shape, "dtype:", vol.dtype)

    v = vol.astype(np.float32, copy=False)
    finite_mask = np.isfinite(v)

    if finite_mask.any():
        vmin = np.percentile(v[finite_mask], 1.0)
        vmax = np.percentile(v[finite_mask], 99.0)
        vol_disp = np.clip((v - vmin) / (vmax - vmin + 1e-8), 0, 1.0)
    else:
        vol_disp = v

    # If a brain mask is provided, apply it
    if brain_mask is not None:
        vol_disp = vol_disp * brain_mask

    def view_slice(z=0, thr=0.5):
        plt.figure(figsize=(6, 6))

        slice_img = vol_disp[:, :, z].T

        # Threshold mask
        roi_mask = slice_img > thr

        # Show both image and ROI overlay
        plt.imshow(slice_img, cmap="gray", origin="lower", interpolation="nearest")
        plt.imshow(np.ma.masked_where(~roi_mask, roi_mask), 
                   cmap="autumn", alpha=0.5, origin="lower", interpolation="nearest")
        plt.axis("off")
        plt.title(f"Slice z={z}, threshold={thr:.2f}")
        plt.show()

    _ = interact(
        view_slice,
        z=widgets.IntSlider(min=0, max=int(vol_disp.shape[2]) - 1, step=1, value=int(vol_disp.shape[2]) // 2),
        thr=widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.5)
    )


# draw nifti_slices, with a timeline slider instead of threshold
def draw_nifti_slices_with_time(imgs, brain_mask=None, normalize='global', percentiles=(1, 99),
                                cmap='gray', transpose_xy=True, overlay_cmap='autumn'):
    """
    Interactive viewer for 4D volume (x,y,z,t) with time, z, and threshold sliders.

    Parameters
      imgs          : 4D numpy array (X,Y,Z,T)
      brain_mask    : optional 3D boolean mask (X,Y,Z)
      normalize     : 'global' or 'per_time'
      percentiles   : (low, high) for intensity clipping
      cmap          : base colormap
      transpose_xy  : transpose slice for display
      overlay_cmap  : colormap for threshold overlay
    """
    assert imgs.ndim == 4, "imgs must be 4D (x,y,z,t)"
    X, Y, Z, T = imgs.shape
    data = imgs.astype(np.float32, copy=False)

    if brain_mask is not None:
        assert brain_mask.shape == (X, Y, Z)
        data = data * brain_mask[..., None]

    finite_mask = np.isfinite(data)
    if not finite_mask.any():
        raise ValueError("No finite voxels found.")

    low_p, high_p = percentiles
    if normalize == 'global':
        vmin = np.percentile(data[finite_mask], low_p)
        vmax = np.percentile(data[finite_mask], high_p)
        vmax = vmin + 1e-8 if vmax <= vmin else vmax
        normed = np.clip((data - vmin) / (vmax - vmin), 0, 1)
    else:
        normed = np.empty_like(data)
        for ti in range(T):
            frame = data[..., ti]
            fm = np.isfinite(frame)
            if fm.any():
                vmin = np.percentile(frame[fm], low_p)
                vmax = np.percentile(frame[fm], high_p)
                vmax = vmin + 1e-8 if vmax <= vmin else vmax
                normed[..., ti] = np.clip((frame - vmin) / (vmax - vmin), 0, 1)
            else:
                normed[..., ti] = frame

    # Initial indices
    t0, z0 = 0, Z // 2
    base_slice = normed[:, :, z0, t0]
    if transpose_xy:
        base_slice = base_slice.T

    fig, ax = plt.subplots(figsize=(5, 5))
    img_artist = ax.imshow(base_slice, cmap=cmap,
                           origin='lower' if transpose_xy else 'upper',
                           interpolation='nearest')

    # Initial threshold overlay (empty until first update)
    roi_mask = base_slice > 0.5
    overlay_artist = ax.imshow(
        np.ma.masked_where(~roi_mask, roi_mask),
        cmap=overlay_cmap, alpha=0.5,
        origin='lower' if transpose_xy else 'upper',
        interpolation='nearest'
    )

    ax.set_title(f"t={t0}/{T-1}, z={z0}/{Z-1}, thr=0.50")
    ax.axis('off')
    plt.tight_layout()

    # Sliders
    t_slider  = widgets.IntSlider(min=0, max=T-1, step=1, value=t0, description='Time')
    z_slider  = widgets.IntSlider(min=0, max=Z-1, step=1, value=z0, description='Z')
    thr_slider = widgets.FloatSlider(min=0.0, max=1.0, step=0.01, value=0.5, description='Thr')

    def update(_):
        t = t_slider.value
        z = z_slider.value
        thr = thr_slider.value
        sl = normed[:, :, z, t]
        if transpose_xy:
            sl = sl.T
        img_artist.set_data(sl)

        mask = sl > thr
        overlay_artist.set_data(np.ma.masked_where(~mask, mask))
        ax.set_title(f"t={t}/{T-1}, z={z}/{Z-1}, thr={thr:.2f}")
        fig.canvas.draw_idle()

    t_slider.observe(update, names='value')
    z_slider.observe(update, names='value')
    thr_slider.observe(update, names='value')

    ui = widgets.VBox([
        widgets.HBox([t_slider, z_slider, thr_slider]),
        fig.canvas
    ])
    display(ui)
    return {
        "figure": fig,
        "time_slider": t_slider,
        "z_slider": z_slider,
        "thr_slider": thr_slider,
        "image_artist": img_artist,
        "overlay_artist": overlay_artist
    }


# now given three 2D image, generate predict + gt + error to form a bigger image
def visualize_prediction_vs_groundtruth(pred_img, gt_img, vmin=0, vmax=1):
    assert pred_img.shape == gt_img.shape, "Prediction and ground truth images must have the same shape."
    # first all shrink to 0-1 range
    img_max = max(pred_img.max(), gt_img.max())
    img_min = min(pred_img.min(), gt_img.min())
    pred_img = (pred_img - img_min) / (img_max - img_min + 1e-8)
    gt_img = (gt_img - img_min) / (img_max - img_min + 1e-8)
    # Compute absolute error 
    error_img = np.abs(pred_img - gt_img)
    # Stack images just horizontally
    stacked = np.vstack((gt_img, pred_img, error_img))
    # Clip values for visualization
    stacked = np.clip(stacked, vmin, vmax)
    return stacked