from __future__ import annotations

from typing import Tuple, Dict

import numpy as np
import pandas as pd


def compute_standardization_params(X: pd.DataFrame) -> Dict[str, np.ndarray]:
    """
    Compute mean and std (standard deviation) for each feature.
    """
    means = X.mean(axis=0).to_numpy()
    stds = X.std(axis=0, ddof=0).to_numpy()

    return {
        "means": means,
        "stds": stds,
        "columns": X.columns.to_list(),
    }


def apply_standardization(
    X: pd.DataFrame,
    params: Dict[str, np.ndarray],
) -> pd.DataFrame:
    """
    Apply standardization using precomputed means and stds.
    """
    X_arr = X.to_numpy().astype(float)
    means = params["means"]
    stds = params["stds"]

    # avoid division by zero
    stds_safe = np.where(stds == 0, 1.0, stds)

    X_scaled = (X_arr - means) / stds_safe

    return pd.DataFrame(X_scaled, columns=params["columns"], index=X.index)
