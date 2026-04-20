from __future__ import annotations

from io import BytesIO
from pathlib import Path
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


_PHASE_KEY_ALIASES = {
    "init_c_data": "cinit",
    "init_c_denoise_data": "cdenoise",
    "ad_pde_p": "ppde",
    "ad_pde_p_k": "pk",
    "joint_ad_pde+joint_data": "joint",
    "joint_ad_pde_joint_data": "joint",
}

_IMAGE_KEY_ALIASES = {
    "val_C_compare": "c",
    "val_c_compare": "c",
    "val_v_quiver": "vq",
    "val_v_mag_slices": "vmag",
    "val_adv_diff_step": "adv",
    "val_k_slices": "k",
    "val_p_slices": "p",
    "val_v_hist": "vh",
    "val_k_hist": "kh",
    "val_p_hist": "ph",
}


def _sanitize_key_token(value: Any) -> str:
    token = str(value).strip().strip("/")
    if not token:
        return ""
    token = token.replace("/", "-").replace("_", "-").replace("+", "-").lower()
    token = re.sub(r"[^0-9a-z.\-]+", "-", token)
    token = re.sub(r"-{2,}", "-", token)
    return token.strip(".-")


def _shorten_token(token: str, max_len: int = 24) -> str:
    token_str = _sanitize_key_token(token)
    if not token_str:
        return ""
    return token_str[:max_len].rstrip(".-")


def compact_image_series_key(active_phase: Any, image_key: Any) -> str:
    phase_raw = str(active_phase or "").strip()
    key_raw = str(image_key or "").strip()

    phase_token = _PHASE_KEY_ALIASES.get(phase_raw)
    if phase_token is None:
        phase_token = _PHASE_KEY_ALIASES.get(_sanitize_key_token(phase_raw), "")
    phase_token = _shorten_token(phase_token or phase_raw, max_len=12)

    key_token = _IMAGE_KEY_ALIASES.get(key_raw)
    if key_token is None:
        key_token = _IMAGE_KEY_ALIASES.get(_sanitize_key_token(key_raw))
    key_token = _shorten_token(key_token or key_raw, max_len=12)

    parts = [part for part in (phase_token, key_token) if part]
    if not parts:
        raise ValueError("image_key must be a non-empty string for comparable image-series logging.")
    return "-".join(parts)


def _phase_scope(logger: Any) -> str:
    return _shorten_token(
        _PHASE_KEY_ALIASES.get(getattr(logger, "active_phase", "") or "", "")
        or (getattr(logger, "active_phase", "") or ""),
        max_len=12,
    )


def _scoped_key(logger: Any, key: str, prefix: str) -> str:
    del prefix
    return compact_image_series_key(getattr(logger, "active_phase", "") or "", key)


def validation_epoch_step(module: Any) -> int:
    """Return a 1-based epoch index for validation artifact logging."""
    return max(1, int(getattr(module, "current_epoch", 0)) + 1)


def should_log_validation_artifacts(module: Any) -> bool:
    """Skip expensive validation artifacts during Lightning sanity checks."""
    trainer = getattr(module, "trainer", None)
    return not bool(getattr(trainer, "sanity_checking", False))


def log_image_artifact(
    logger: Any,
    image: np.ndarray,
    image_key: str,
    step: int,
) -> None:
    """Log an MLflow keyed image-series artifact.

    MLflow persists keyed image logs as run artifacts under `artifacts/images/`
    while also exposing them through its image-series UI.
    """
    if logger is None or logger is False:
        return
    experiment = getattr(logger, "experiment", None)
    run_id = getattr(logger, "run_id", None)
    if experiment is None or run_id is None:
        return
    image_key_str = _scoped_key(logger, image_key, prefix="validation")
    if not image_key_str:
        raise ValueError("image_key must be a non-empty string for comparable image-series logging.")
    image_array = np.asarray(image)
    run_id = logger.run_id
    step_i = int(step)

    logger.experiment.log_image(
        run_id=run_id,
        image=image_array,
        key=image_key_str,
        step=step_i,
        synchronous=True,
    )


def log_histogram_artifact(
    logger: Any,
    values: np.ndarray,
    hist_key: str,
    step: int,
    bins: int = 64,
) -> None:
    """Render a histogram and log it as an MLflow keyed image artifact."""
    if logger is None or logger is False:
        return
    hist_key_str = str(hist_key).strip()
    if not hist_key_str:
        raise ValueError("hist_key must be a non-empty string.")

    values_np = np.asarray(values).reshape(-1)
    if values_np.size == 0:
        return
    values_np = values_np[np.isfinite(values_np)]
    if values_np.size == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.hist(values_np, bins=int(bins), color="#2f5d8a", alpha=0.9)
    ax.set_title(hist_key_str)
    ax.set_xlabel("Value")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)

    image = plt.imread(buffer, format="png")
    # MLflow accepts HWC images; drop alpha channel if present.
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]
    log_image_artifact(logger=logger, image=image, image_key=hist_key_str, step=step)


def log_file_artifact(
    logger: Any,
    local_path: str,
    artifact_path: str | None = None,
) -> None:
    """Persist a local file as an MLflow artifact for the active run."""
    if logger is None or logger is False:
        return
    experiment = getattr(logger, "experiment", None)
    run_id = getattr(logger, "run_id", None)
    if experiment is None or run_id is None:
        return

    path_obj = Path(local_path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Artifact file not found: {local_path}")

    artifact_path_str = str(artifact_path).strip().strip("/") if artifact_path else None
    experiment.log_artifact(run_id=run_id, local_path=str(path_obj), artifact_path=artifact_path_str)


def log_text_artifact(
    logger: Any,
    text: str,
    text_key: str,
    step: int,
) -> None:
    """Persist text as an MLflow run artifact when supported by client."""
    if logger is None or logger is False:
        return
    text_key_str = str(text_key).strip().strip("/")
    if not text_key_str:
        return
    run_id = logger.run_id
    artifact_file = f"{text_key_str}/step_{int(step):08d}.txt"

    log_text = getattr(logger.experiment, "log_text", None)
    if callable(log_text):
        log_text(run_id=run_id, text=str(text), artifact_file=artifact_file)
