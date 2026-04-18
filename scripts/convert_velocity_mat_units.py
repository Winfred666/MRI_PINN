#!/usr/bin/env python3
import argparse
from fractions import Fraction
from pathlib import Path

import numpy as np
import scipy.io as sio


def parse_fraction(value: str) -> float:
    text = str(value).strip()
    if not text:
        raise ValueError("Expected a non-empty numeric value.")
    try:
        return float(Fraction(text))
    except Exception:
        return float(text)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a saved predict_velocity.mat payload into an explicit dx/dt convention. "
            "The conversion starts from velocity_3d_cell_min (voxels/minute)."
        )
    )
    parser.add_argument("input_mat", type=str, help="Path to predict_velocity.mat")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output .mat path. Defaults to <input>_dx1over32_dt1.mat",
    )
    parser.add_argument(
        "--frame-delta-minutes",
        type=float,
        default=4.0,
        help="Physical minutes represented by one adjacent-frame step in the source data.",
    )
    parser.add_argument(
        "--dx",
        type=str,
        default="1/32",
        help="Target spatial step, accepts decimals or simple fractions like 1/32.",
    )
    parser.add_argument(
        "--dt",
        type=str,
        default="1",
        help="Target temporal step, accepts decimals or simple fractions like 1 or 1/2.",
    )
    args = parser.parse_args()

    input_path = Path(args.input_mat)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input mat file not found: {input_path}")

    dx = parse_fraction(args.dx)
    dt = parse_fraction(args.dt)
    if dt == 0.0:
        raise ValueError("Target dt must be non-zero.")

    payload = sio.loadmat(input_path)

    if "velocity_3d_cell_min" in payload:
        velocity_3d_cell_min = np.asarray(payload["velocity_3d_cell_min"], dtype=np.float32)
    else:
        required = ["vx_mm_min", "vy_mm_min", "vz_mm_min", "pixdim"]
        missing = [key for key in required if key not in payload]
        if missing:
            raise KeyError(
                "Input mat is missing velocity_3d_cell_min and fallback fields: "
                + ", ".join(missing)
            )
        pixdim = np.asarray(payload["pixdim"]).reshape(-1).astype(np.float32)
        vx_mm_min = np.asarray(payload["vx_mm_min"], dtype=np.float32)
        vy_mm_min = np.asarray(payload["vy_mm_min"], dtype=np.float32)
        vz_mm_min = np.asarray(payload["vz_mm_min"], dtype=np.float32)
        velocity_3d_cell_min = np.stack(
            [vx_mm_min / pixdim[0], vy_mm_min / pixdim[1], vz_mm_min / pixdim[2]],
            axis=-1,
        )

    velocity_3d_cell_frame = velocity_3d_cell_min * float(args.frame_delta_minutes)
    velocity_3d_dxdt = velocity_3d_cell_frame * (dx / dt)

    dx_label = str(args.dx).replace("/", "over").replace(".", "p")
    dt_label = str(args.dt).replace("/", "over").replace(".", "p")
    default_output = input_path.with_name(f"{input_path.stem}_dx{dx_label}_dt{dt_label}.mat")
    output_path = Path(args.output) if args.output else default_output

    converted_payload = dict(payload)
    converted_payload.update(
        {
            "velocity_3d_cell_frame": velocity_3d_cell_frame.astype(np.float32),
            "vx_cell_frame": velocity_3d_cell_frame[..., 0].astype(np.float32),
            "vy_cell_frame": velocity_3d_cell_frame[..., 1].astype(np.float32),
            "vz_cell_frame": velocity_3d_cell_frame[..., 2].astype(np.float32),
            "velocity_3d_dxdt": velocity_3d_dxdt.astype(np.float32),
            "vx_dxdt": velocity_3d_dxdt[..., 0].astype(np.float32),
            "vy_dxdt": velocity_3d_dxdt[..., 1].astype(np.float32),
            "vz_dxdt": velocity_3d_dxdt[..., 2].astype(np.float32),
            "conversion_source": "velocity_3d_cell_min",
            "conversion_formula": (
                "velocity_3d_dxdt = velocity_3d_cell_min * frame_delta_minutes * dx / dt"
            ),
            "frame_delta_minutes": float(args.frame_delta_minutes),
            "target_dx": float(dx),
            "target_dt": float(dt),
        }
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sio.savemat(output_path, converted_payload)
    print(f"Saved converted velocity mat to {output_path}")
    print(
        "Formula: velocity_3d_dxdt = velocity_3d_cell_min * "
        f"{args.frame_delta_minutes} * {dx} / {dt}"
    )


if __name__ == "__main__":
    main()
