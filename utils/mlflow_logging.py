from __future__ import annotations

from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


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
    image_key_str = str(image_key).strip()
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
