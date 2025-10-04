import numpy as np

import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import interact
from IPython.display import display

# 1. visualize 3D velocity field quiver with interactive widget or window
def draw_3d_quiver(X, Y, Z, vx, vy, vz, stride=3, length_scale=0.25, cmap='RdYlBu_r', normalize_vectors=False, ax=None, show=True, return_mappable=False):
    """
    Colored 3D quiver: arrow length + color encode speed magnitude.
    If ax is provided, update the existing axes instead of creating a new figure.
    Parameters
      show: if True and a new figure is created (ax is None), call plt.show().
      return_mappable: if True, also return a ScalarMappable for creating a colorbar (independent of artist alpha).
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
    if mag_max <= mag_min:
        mag_max = mag_min + 1.0
    mag_norm = (mag - mag_min) / (mag_max - mag_min)  # 0..1 for colormap

    if normalize_vectors:
        nonzero = mag > 0
        Uf[nonzero] /= mag[nonzero]
        Vf[nonzero] /= mag[nonzero]
        Wf[nonzero] /= mag[nonzero]
    else:
        max_m = mag.max() if mag.size else 1.0
        if max_m > 0:
            Uf /= max_m
            Vf /= max_m
            Wf /= max_m

    # Apply global length scale
    Uf *= length_scale
    Vf *= length_scale
    Wf *= length_scale

    colors = plt.get_cmap(cmap)(mag_norm)

    created_new_ax = False
    if ax is None:
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        created_new_ax = True
    else:
        fig = ax.figure
        ax.clear()

    # Quiver (colored arrows)
    ax.quiver(Xf, Yf, Zf, Uf, Vf, Wf, colors=colors, length=1.0, normalize=False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Velocity Field (colored by |v|)')
    ax.set_box_aspect([1,1,1])
    plt.tight_layout()

    # Independent mappable for colorbar so alpha is not affected
    if return_mappable:
        norm = plt.Normalize(vmin=mag_min, vmax=mag_max)
        mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        mappable.set_array(mag)
    else:
        mappable = None

    if created_new_ax and show:
        plt.show()
    return (fig, ax, mappable) if return_mappable else (fig, ax)

def interactive_quiver(vx, vy, vz, pixdim, default_elev=None, default_azim=None):
    """
    Interactive 3D quiver plot with sliders.
    Parameters:
        vx, vy, vz: velocity components
        pixdim: pixel dimensions for scaling
        default_elev: default elevation angle (degrees)
        default_azim: default azimuth angle (degrees)
    """
    X, Y, Z = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]), np.arange(vx.shape[2]), indexing='ij')
    X = X * pixdim[0]
    Y = Y * pixdim[1]
    Z = Z * pixdim[2]

    # Create figure and axes once
    fig, ax = plt.subplots(figsize=(9, 7), subplot_kw={'projection': '3d'})
    cbar = None  # Will be added on first draw
    stored_mappable = None

    def _show(stride=3, length_scale=25, normalize_vectors=False):
        nonlocal default_elev, default_azim, stored_mappable, cbar
        # Save current view angles (if any)
        elev = ax.elev
        azim = ax.azim
        # Update the plot (request mappable)
        fig_ret = draw_3d_quiver(
            X, Y, Z, vx, vy, vz,
            stride=stride,
            length_scale=length_scale/100.0,
            normalize_vectors=normalize_vectors,
            ax=ax,
            show=False,
            return_mappable=True
        )
        _, _, mappable = fig_ret
        stored_mappable = mappable

        # Apply defaults if provided (overrides saved values)
        if default_elev is not None:
            elev = default_elev
            default_elev = None  # Only apply once
        if default_azim is not None:
            azim = default_azim
            default_azim = None  # Only apply once

        # Restore/set view angles
        ax.view_init(elev=elev, azim=azim)

        # Add colorbar only once
        if cbar is None and stored_mappable is not None:
            cbar = fig.colorbar(stored_mappable, ax=ax, shrink=0.65, pad=0.1)
            cbar.set_label('|v| (magnitude)')

        fig.canvas.draw_idle()

    interact(
        _show,
        stride=widgets.IntSlider(min=1, max=10, step=1, value=3, description='Stride'),
        length_scale=widgets.IntSlider(min=1, max=100, step=1, value=25, description='Len %'),
        normalize_vectors=widgets.Checkbox(value=False, description='Uniform length')
    )

    # Display the initial plot with defaults applied
    _show()


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
def draw_nifti_slices_with_time(pred_imgs, gt_imgs=None, brain_mask=None, normalize='global', percentiles=(1, 99),
                                cmap='gray', transpose_xy=True, overlay_cmap='autumn'):
    """
    Interactive viewer for 4D volume (x,y,z,t) with time, z, and threshold sliders.

    Parameters
      pred_imgs          : 4D numpy array (X,Y,Z,T)
      gt_imgs       : 4D numpy array (X,Y,Z,T) for ground truth
      brain_mask    : optional 3D boolean mask (X,Y,Z)
      normalize     : 'global' or 'per_time'
      percentiles   : (low, high) for intensity clipping
      cmap          : base colormap
      transpose_xy  : transpose slice for display
      overlay_cmap  : colormap for threshold overlay
    """

    # Now as predict images and gt images are x,y,z,t,
    # with same z and t, we hstack (x,y) to form a bigger image
    X, Y, Z, T = pred_imgs.shape
    if brain_mask is not None:
        assert brain_mask.shape == (X, Y, Z)
        pred_imgs = pred_imgs * brain_mask[..., None]
    if gt_imgs is not None:
        imgs = np.concatenate([gt_imgs, pred_imgs, np.abs(pred_imgs - gt_imgs)], axis=0)
    else:
        imgs = pred_imgs
    X, Y, Z, T = imgs.shape
    assert imgs.ndim == 4, "imgs must be 4D (x,y,z,t)"
    
    data = imgs.astype(np.float32, copy=False)

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

# Add a fixed-view quiver exporter returning an RGB array
def fixed_quiver_image(vx, vy, vz, pixdim, stride=2, length_scale=0.8, elev=-62.76, azim=-10.87,
                       cmap='RdYlBu_r', normalize_vectors=False, figsize=(7, 6), dpi=100,
                       add_colorbar=True, close_fig=True):
    """
    Render a 3D quiver plot at a fixed view and return it as an RGB (H,W,3) uint8 array.
    Uses separate ScalarMappable so colorbar always shows colors.
    """
    X, Y, Z = np.meshgrid(np.arange(vx.shape[0]), np.arange(vx.shape[1]), np.arange(vx.shape[2]), indexing='ij')
    X = X * pixdim[0]; Y = Y * pixdim[1]; Z = Z * pixdim[2]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')

    fig_ret = draw_3d_quiver(X, Y, Z, vx, vy, vz,
                   stride=stride,
                   length_scale=length_scale,
                   cmap=cmap,
                   normalize_vectors=normalize_vectors,
                   ax=ax,
                   show=False,
                   return_mappable=True)
    _, _, mappable = fig_ret

    ax.view_init(elev=elev, azim=azim)

    if add_colorbar and mappable is not None:
        cb = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.1)
        cb.set_label('|v| (magnitude)')

    # Robust figure-to-RGB extraction
    rgb = None
    try:
        fig.canvas.draw()
        if hasattr(fig.canvas, "tostring_rgb"):
            w, h = fig.canvas.get_width_height()
            buf = fig.canvas.tostring_rgb()
            rgb = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        elif hasattr(fig.canvas, "buffer_rgba"):
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            rgb = buf[..., :3].copy()
        else:
            raise AttributeError("No direct RGB buffer method on canvas.")
    except Exception:
        import io
        from PIL import Image
        bio = io.BytesIO()
        fig.savefig(bio, format='png', dpi=fig.dpi, bbox_inches='tight')
        bio.seek(0)
        rgb = np.array(Image.open(bio).convert("RGB"))

    if close_fig:
        plt.close(fig)

    return rgb
