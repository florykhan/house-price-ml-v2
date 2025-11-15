from __future__ import annotations

from typing import Dict

import numpy as np


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """
    Compute evaluation metrics for regression:
    - MAE
    - RMSE
    - R^2
    """

    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    # R^2 score: 1 - (SS_res / SS_tot)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }
