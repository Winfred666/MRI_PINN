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
def draw_nifti_slices_with_threshold(img, brain_mask=None, slice_along_axis='z'):
    """
    Interactive viewer for a 3D volume with slice and threshold sliders.

    Parameters:
        img (np.ndarray): 3D or 4D numpy array. If 4D, the first time point is used.
        brain_mask (np.ndarray, optional): 3D boolean mask.
        slice_along_axis (str): The axis to slice along ('x', 'y', or 'z').
    """
    if slice_along_axis not in ['x', 'y', 'z']:
        raise ValueError("slice_along_axis must be one of 'x', 'y', or 'z'")

    # Get array (memory-mapped if possible)
    data = img
    vol = data[..., 0] if data.ndim == 4 else data
    print("Volume shape:", vol.shape, "dtype:", vol.dtype)

    v = vol.astype(np.float32, copy=False)
    finite_mask = np.isfinite(v)
    
    vmin, vmax = 0, 1 # Default values
    if finite_mask.any():
        vmin = np.percentile(v[finite_mask], 0.0)
        vmax = np.percentile(v[finite_mask], 100.0)
        vmax = vmin + 1e-8 if vmax <= vmin else vmax
        vol_disp = np.clip((v - vmin) / (vmax - vmin), 0, 1.0)
    else:
        vol_disp = v

    # If a brain mask is provided, apply it to the display volume
    if brain_mask is not None:
        assert brain_mask.shape == vol.shape, "Mask shape must match volume shape"
        vol_disp = vol_disp * brain_mask

    # --- Axis-dependent setup ---
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    slice_axis_idx = axis_map[slice_along_axis]
    slice_max = vol.shape[slice_axis_idx] - 1

    def view_slice(slice_idx=0, thr=0.5):
        plt.figure(figsize=(6, 6))

        # Create a slicer for the chosen axis
        slicer = [slice(None)] * 3
        slicer[slice_axis_idx] = slice_idx
        slicer = tuple(slicer)

        # Display image is normalized for consistent colormapping
        slice_disp_img = vol_disp[slicer].T
        # Original data slice for absolute thresholding
        slice_orig_img = v[slicer].T

        # Threshold mask is now on original data
        roi_mask = slice_orig_img > thr

        # Show both image and ROI overlay
        plt.imshow(slice_disp_img, cmap="gray", origin="lower", interpolation="nearest")
        plt.imshow(np.ma.masked_where(~roi_mask, roi_mask), 
                   cmap="autumn", alpha=0.5, origin="lower", interpolation="nearest")
        plt.axis("off")
        plt.title(f"Slice {slice_along_axis}={slice_idx}, threshold={thr:.2f}")
        plt.show()

    # Create the interactive widget for the chosen slice axis
    slice_slider = widgets.IntSlider(
        min=0, max=slice_max, step=1, value=slice_max // 2,
        description=f'Slice {slice_along_axis.upper()}'
    )
    
    interact_kwargs = {
        'slice_idx': slice_slider,
        'thr': widgets.FloatSlider(min=vmin, max=vmax, step=(vmax-vmin)/100, value=(vmin+vmax)/2, description='Abs Thr')
    }

    _ = interact(view_slice, **interact_kwargs)


# draw nifti_slices, with a timeline slider instead of threshold
def draw_nifti_slices_with_time(pred_imgs, gt_imgs=None, brain_mask=None, normalize='global', percentiles=(1, 99.5),
                                cmap='gray', transpose_xy=True, overlay_cmap='autumn', slice_along_axis='z'):
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
      slice_along_axis: 'x', 'y', or 'z'. The axis to slice along.
    """
    if slice_along_axis not in ['x', 'y', 'z']:
        raise ValueError("slice_along_axis must be one of 'x', 'y', or 'z'")

    X, Y, Z, T = pred_imgs.shape
    if brain_mask is not None:
        assert brain_mask.shape == (X, Y, Z)
        pred_imgs = pred_imgs * brain_mask[..., None]
    
    # Keep pred and gt separate until slicing
    data_pred = pred_imgs.astype(np.float32, copy=False)
    data_gt = gt_imgs.astype(np.float32, copy=False) if gt_imgs is not None else None

    # Determine normalization range from all available data
    all_data_for_norm = np.concatenate([d for d in [data_pred, data_gt] if d is not None], axis=0)
    finite_mask = np.isfinite(all_data_for_norm)
    if not finite_mask.any():
        raise ValueError("No finite voxels found.")

    low_p, high_p = percentiles
    if normalize == 'global':
        vmin = np.percentile(all_data_for_norm[finite_mask], low_p)
        vmax = np.percentile(all_data_for_norm[finite_mask], high_p)
    else: # Placeholder for per-time, which is more complex with combined images
        vmin, vmax = np.percentile(all_data_for_norm[finite_mask], [low_p, high_p])
    vmax = vmin + 1e-8 if vmax <= vmin else vmax

    # --- Axis-dependent setup ---
    axis_map = {'x': 0, 'y': 1, 'z': 2}
    slice_axis_idx = axis_map[slice_along_axis]
    
    # Determine slice dimensions and slider range
    slice_dims = [X, Y, Z]
    slice_max = slice_dims.pop(slice_axis_idx) - 1
    slice_dim_labels = ['X', 'Y', 'Z']
    slice_dim_labels.pop(slice_axis_idx)

    # Initial indices
    t0 = 0
    slice0 = slice_max // 2

    def get_slice(data_4d, t, slice_idx):
        if data_4d is None: return None
        # Use slicing to extract the 2D plane
        slicer = [slice(None)] * 4
        slicer[slice_axis_idx] = slice_idx
        slicer[3] = t
        return data_4d[tuple(slicer)]

    def get_combined_slice(t, slice_idx):
        pred_slice = get_slice(data_pred, t, slice_idx)
        if data_gt is not None:
            gt_slice = get_slice(data_gt, t, slice_idx)
            err_slice = np.abs(pred_slice - gt_slice)
            # Stack horizontally
            return np.hstack([gt_slice, pred_slice, err_slice])
        return pred_slice

    # Get initial slice for setup
    base_slice_original = get_combined_slice(t0, slice0)
    base_slice_normed = np.clip((base_slice_original - vmin) / (vmax - vmin), 0, 1)
    
    if transpose_xy:
        base_slice_normed = base_slice_normed.T

    fig, ax = plt.subplots(figsize=(8, 4))
    img_artist = ax.imshow(base_slice_normed, cmap=cmap,
                           origin='lower' if transpose_xy else 'upper',
                           interpolation='nearest')

    # Initial threshold overlay
    roi_mask = base_slice_original > ((vmin + vmax) / 2)
    if transpose_xy: roi_mask = roi_mask.T
    overlay_artist = ax.imshow(
        np.ma.masked_where(~roi_mask, roi_mask),
        cmap=overlay_cmap, alpha=0.5,
        origin='lower' if transpose_xy else 'upper',
        interpolation='nearest'
    )

    ax.set_title(f"t={t0}/{T-1}, {slice_along_axis}={slice0}/{slice_max}, thr=0.50")
    ax.axis('off')
    plt.tight_layout()

    # Custom formatter to show original data values on hover
    def formatter(x, y):
        t = t_slider.value
        slice_val = slice_slider.value

        col, row = int(x + 0.5), int(y + 0.5)
        
        original_slice = get_combined_slice(t, slice_val)
        if transpose_xy:
            original_slice = original_slice.T

        if 0 <= row < original_slice.shape[0] and 0 <= col < original_slice.shape[1]:
            val = original_slice[row, col]
            return f'({slice_dim_labels[0]}, {slice_dim_labels[1]}) = ({x:.1f}, {y:.1f}), value={val:.4f}'
        else:
            return f'x={x:.1f}, y={y:.1f}'

    ax.format_coord = formatter

    # Sliders
    t_slider  = widgets.IntSlider(min=0, max=T-1, step=1, value=t0, description='Time')
    slice_slider = widgets.IntSlider(min=0, max=slice_max, step=1, value=slice0, description=f'Slice {slice_along_axis.upper()}')
    thr_slider = widgets.FloatSlider(min=vmin, max=vmax, step=(vmax-vmin)/100, value=(vmin+vmax)/2, description='Abs Thr')

    def update(_):
        t = t_slider.value
        slice_idx = slice_slider.value
        thr = thr_slider.value

        sl_original = get_combined_slice(t, slice_idx)
        sl_normed = np.clip((sl_original - vmin) / (vmax - vmin), 0, 1)

        if transpose_xy:
            sl_normed = sl_normed.T
            sl_original = sl_original.T

        img_artist.set_data(sl_normed)
        img_artist.set_extent((0, sl_normed.shape[1], 0, sl_normed.shape[0]))

        mask = sl_original > thr
        overlay_artist.set_data(np.ma.masked_where(~mask, mask))
        overlay_artist.set_extent((0, sl_normed.shape[1], 0, sl_normed.shape[0]))
        
        ax.set_title(f"t={t}/{T-1}, {slice_along_axis}={slice_idx}/{slice_max}, thr={thr:.2f}")
        fig.canvas.draw_idle()

    t_slider.observe(update, names='value')
    slice_slider.observe(update, names='value')
    thr_slider.observe(update, names='value')

    ui = widgets.VBox([
        widgets.HBox([t_slider, slice_slider, thr_slider]),
        fig.canvas
    ])
    display(ui)
    return {
        "figure": fig,
        "time_slider": t_slider,
        "slice_slider": slice_slider,
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
def fixed_quiver_image(vx, vy, vz, pixdim, stride=3, length_scale=0.8, elev=-72.76, azim=-10.87,
                       cmap='RdYlBu_r', normalize_vectors=False, figsize=(7, 6), dpi=100,
                       add_colorbar=True, close_fig=True, zoom=1.25, label="|v| magnitude"):
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

    # Simulated zoom via axis limits
    if zoom > 1.0:
        # Determine center in voxel indices
        cx = (vx.shape[0]-1)/2.0
        cy = (vy.shape[1]-1)/2.0 if vx.ndim == 3 else (vx.shape[1]-1)/2.0
        cz = (vz.shape[2]-1)/2.0
        # Original physical spans
        x_full = (vx.shape[0]-1)*pixdim[0]
        y_full = (vy.shape[1]-1)*pixdim[1]
        z_full = (vz.shape[2]-1)*pixdim[2]
        # Enforce a minimum fraction of span
        frac = 1.0/zoom
        x_half = 0.5 * x_full * frac
        y_half = 0.5 * y_full * frac
        z_half = 0.5 * z_full * frac
        cxp = cx * pixdim[0]
        cyp = cy * pixdim[1]
        czp = cz * pixdim[2]
        ax.set_xlim(max(0, cxp - x_half), min(x_full, cxp + x_half))
        ax.set_ylim(max(0, cyp - y_half), min(y_full, cyp + y_half))
        ax.set_zlim(max(0, czp - z_half), min(z_full, czp + z_half))


    if add_colorbar and mappable is not None:
        cb = fig.colorbar(mappable, ax=ax, shrink=0.65, pad=0.1)
        cb.set_label(label)

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


def draw_colorful_slice_image(slice_data, cmap='viridis', mask=None):
    """
    Renders a 2D data slice into a colorful RGB image array in a headless manner.
    The plot has no axes or title, and a tight colorbar showing the original data range.

    Args:
        slice_data (np.ndarray): The 2D numpy array to plot.
        cmap (str): The name of the matplotlib colormap to use.

    Returns:
        np.ndarray: An RGB (H, W, 3) uint8 array of the resulting image.
    """

    # Handle mask if provided
    if mask is not None:
        # Determine data range for the color bar, should only select in mask
        vmin = slice_data[mask].min()
        vmax = slice_data[mask].max()
        # Mask where mask is False
        slice_data = np.ma.masked_where(~mask, slice_data)
        cmap = plt.get_cmap(cmap).copy()
        cmap.set_bad(color='white')
    else:
        vmin = slice_data.min()
        vmax = slice_data.max()
    # Create figure and axes with a specific size and DPI
    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)

    # Plot the image data. imshow handles mapping values to the colormap via vmin/vmax.
    im = ax.imshow(slice_data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower', interpolation='nearest')

    # Add a colorbar, making it compact
    fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    # Remove axis ticks and labels
    ax.axis('off')
    # Ensure tight layout before rendering
    plt.tight_layout(pad=0)

    # Robust figure-to-RGB extraction (adapted from fixed_quiver_image)
    rgb = None
    try:
        fig.canvas.draw()
        # Try modern, direct buffer access first
        if hasattr(fig.canvas, "buffer_rgba"):
            w, h = fig.canvas.get_width_height()
            buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            rgb = buf[..., :3].copy() # Drop alpha channel
        # Fallback for older versions
        elif hasattr(fig.canvas, "tostring_rgb"):
            w, h = fig.canvas.get_width_height()
            buf = fig.canvas.tostring_rgb()
            rgb = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
        else:
            raise AttributeError("No direct RGB buffer method found on canvas.")
    except Exception:
        # Fallback to saving the figure to an in-memory buffer if direct methods fail
        import io
        from PIL import Image
        bio = io.BytesIO()
        # bbox_inches='tight' is crucial for removing whitespace
        fig.savefig(bio, format='png', dpi=fig.dpi, bbox_inches='tight', pad_inches=0)
        bio.seek(0)
        rgb = np.array(Image.open(bio).convert("RGB"))

    # Close the figure to free up memory
    plt.close(fig)

    return rgb
