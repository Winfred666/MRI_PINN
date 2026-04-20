from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import glob
import re

from utils.path_io import PathInput, to_abs_path


def _resolve_series_root(template_path: Path) -> Path:
    """Return mouse-case root for one indexed NIfTI template path."""
    if template_path.parent.name == "raw" and template_path.parent.parent.exists():
        return template_path.parent.parent
    return template_path.parent


def _load_simpleitk() -> object:
    """Import and return SimpleITK lazily for runtime-only dependency use."""
    try:
        import SimpleITK as sitk
    except ImportError as exc:  # pragma: no cover - runtime dependency
        raise ImportError(
            "ensure_pdce2base_template_from_niigz requires SimpleITK. "
            "Install with: pip install SimpleITK"
        ) from exc
    return sitk


def _run_simpleitk_rigid_registration_in_memory(
    *,
    moving_path: Path,
    fixed_path: Path,
    fixed: object,
    sitk: object,
) -> object:
    """Rigidly register one moving frame onto one fixed frame and return corrected image."""
    if moving_path.resolve() == fixed_path.resolve():
        return fixed

    moving = sitk.ReadImage(str(moving_path), sitk.sitkFloat32)

    initial_transform = sitk.CenteredTransformInitializer(
        fixed,
        moving,
        sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration.SetMetricSamplingStrategy(registration.RANDOM)
    registration.SetMetricSamplingPercentage(0.10)
    registration.SetInterpolator(sitk.sitkLinear)
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=2.0,
        minStep=1e-4,
        numberOfIterations=200,
        relaxationFactor=0.5,
    )
    registration.SetOptimizerScalesFromPhysicalShift()
    registration.SetShrinkFactorsPerLevel([4, 2, 1])
    registration.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    registration.SetInitialTransform(initial_transform, inPlace=False)

    final_transform = registration.Execute(fixed, moving)
    corrected = sitk.Resample(
        moving,
        fixed,
        final_transform,
        sitk.sitkLinear,
        0.0,
        moving.GetPixelID(),
    )
    return corrected


def _sitk_image_to_xyz_array(
    *,
    image: object,
    np_module: object,
    sitk: object,
) -> object:
    """Convert a SimpleITK image to `(X,Y,Z)` numpy array matching nibabel layout."""
    return np_module.transpose(
        sitk.GetArrayFromImage(image),
        (2, 1, 0),
    ).astype(np_module.float32, copy=False)


def _load_aligned_xyz_array(
    *,
    moving_path: Path,
    fixed_path: Path,
    fixed_image: object,
    sitk: object,
    np_module: object,
) -> object:
    """Load one frame, rigidly align to the fixed image, and return `(X,Y,Z)` numpy array."""
    corrected = _run_simpleitk_rigid_registration_in_memory(
        moving_path=moving_path,
        fixed_path=fixed_path,
        fixed=fixed_image,
        sitk=sitk,
    )
    return _sitk_image_to_xyz_array(
        image=corrected,
        np_module=np_module,
        sitk=sitk,
    )


def ensure_pdce2base_template_from_niigz(
    data_template: PathInput,
    frame_indices: Sequence[int],
    *,
    baseline_glob: str = "**/*baseline*.nii.gz",
    output_dir: Optional[PathInput] = None,
    output_name_template: str = "pdce2base_{:04d}.nii",
    baseline_epsilon: float = 1.0e-6,
) -> str:
    """Ensure raw `.nii.gz` dynamic frames are realigned and converted into `new_DCE` `.nii`."""
    template_str = str(data_template)
    template_path = Path(template_str)
    if template_path.suffix.lower() != ".gz" or not template_str.lower().endswith(".nii.gz"):
        return template_str
    if not frame_indices:
        raise ValueError("frame_indices must not be empty")
    if float(baseline_epsilon) < 0.0:
        raise ValueError(f"baseline_epsilon must be >= 0.0, got {baseline_epsilon}")

    try:
        import matplotlib
        import nibabel as nib  # type: ignore
        import numpy as np
    except ImportError as exc:  # pragma: no cover - notebook/runtime dependency
        raise ImportError(
            "ensure_pdce2base_template_from_niigz requires matplotlib, nibabel, and numpy."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    resolved_template = to_abs_path(template_path)
    mouse_root = _resolve_series_root(resolved_template)
    search_root = resolved_template.parent if resolved_template.parent.name == "raw" else mouse_root
    resolved_output_dir = (
        to_abs_path(output_dir)
        if output_dir is not None
        else mouse_root / "new_DCE"
    )
    output_template_path = resolved_output_dir / output_name_template

    target_paths = [to_abs_path(str(resolved_template).format(int(i))) for i in frame_indices]
    missing_targets = [p for p in target_paths if not p.exists()]
    if missing_targets:
        raise FileNotFoundError(f"Missing source NIfTI frames: {missing_targets[:3]}")

    baseline_candidates = sorted(
        Path(p).resolve() for p in glob.glob(str(search_root / baseline_glob), recursive=True)
    )
    baseline_paths = [p for p in baseline_candidates if p.is_file()]
    if not baseline_paths:
        raise FileNotFoundError(
            f"No baseline .nii.gz found using pattern '{baseline_glob}' under {search_root}"
        )

    sitk = _load_simpleitk()
    first_baseline_path = baseline_paths[0]
    fixed_image = sitk.ReadImage(str(first_baseline_path), sitk.sitkFloat32)
    fixed_nifti = nib.load(str(first_baseline_path))

    baseline_stack = [
        np.nan_to_num(
            _load_aligned_xyz_array(
                moving_path=path,
                fixed_path=first_baseline_path,
                fixed_image=fixed_image,
                sitk=sitk,
                np_module=np,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        for path in baseline_paths
    ]
    baseline_mean = np.mean(np.stack(baseline_stack, axis=0), axis=0, dtype=np.float32)

    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    original_signal_sums: list[float] = []
    normalized_signal_sums: list[float] = []
    for frame_idx, src_path in zip(frame_indices, target_paths):
        src_data = np.nan_to_num(
            _load_aligned_xyz_array(
                moving_path=src_path,
                fixed_path=first_baseline_path,
                fixed_image=fixed_image,
                sitk=sitk,
                np_module=np,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        original_signal_sums.append(float(src_data.sum(dtype=np.float64)))
        delta = ((src_data - baseline_mean) / (baseline_mean + float(baseline_epsilon))).astype(
            np.float32,
            copy=False,
        )
        normalized_signal_sums.append(float(delta.sum(dtype=np.float64)))
        out_img = nib.Nifti1Image(delta, fixed_nifti.affine, fixed_nifti.header.copy())
        out_img.set_data_dtype(np.float32)
        out_path = resolved_output_dir / output_name_template.format(int(frame_idx))
        nib.save(out_img, str(out_path))

    frame_axis = [int(frame_idx) for frame_idx in frame_indices]
    fig, ax_left = plt.subplots(figsize=(9, 5), dpi=220)
    ax_right = ax_left.twinx()
    left_line = ax_left.plot(
        frame_axis,
        original_signal_sums,
        color="#005f73",
        linewidth=2.0,
        marker="o",
        label="aligned_original_signal_sum",
    )[0]
    right_line = ax_right.plot(
        frame_axis,
        normalized_signal_sums,
        color="#ae2012",
        linewidth=2.0,
        marker="s",
        label="baseline_normalized_signal_sum",
    )[0]
    ax_left.set_xlabel("Frame")
    ax_left.set_ylabel("Original Signal Sum", color="#005f73")
    ax_right.set_ylabel("Baseline-Normalized Signal Sum", color="#ae2012")
    ax_left.tick_params(axis="y", labelcolor="#005f73")
    ax_right.tick_params(axis="y", labelcolor="#ae2012")
    ax_left.set_title("Total Mass Changes vs Frames")
    ax_left.grid(True, alpha=0.3)
    ax_left.legend(
        handles=[left_line, right_line],
        labels=[left_line.get_label(), right_line.get_label()],
        loc="best",
    )
    fig.tight_layout()
    fig.savefig(resolved_output_dir / "mass_conservation_analysis.png")
    plt.close(fig)

    return str(output_template_path)


__all__ = [
    "ensure_pdce2base_template_from_niigz",
]
