"""Brightness-based placeholder estimator for testing the pipeline."""

import numpy as np


def count(image: np.ndarray) -> float:
    """Count proportional to image darkness; not a crowd counter."""
    return float(500.0 * (1.0 - image.mean() / 255.0))
