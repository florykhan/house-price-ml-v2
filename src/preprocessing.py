from __future__ import annotations

from typing import Tuple, Dict

import numpy as np
import pandas as pd


def compute_standardization_params(X: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Compute mean and std (standard deviation) for each feature.
    """
    means = X.mean()
    stds = X.std(ddof=0)

    return {
        "means": means,
        "stds": stds,
        "columns": X.columns.to_list(),
    }


def apply_standardization(
    X: pd.DataFrame,
    params: Dict[str, pd.Series],
) -> pd.DataFrame:
    """
    Apply standardization using precomputed means and stds.
    """
    # 1. Align columns to training columns
    train_cols = params["columns"]
    X = X.reindex(columns=train_cols)

    # 2. Fill missing values BEFORE scaling
    X = X.fillna(0)   # or better: X.fillna(means), but 0 is stable for ratios

    means = params["means"]
    stds = params["stds"].replace(0, 1.0) # avoid divide-by-zero

    # 3. Apply scaling
    X_scaled = (X - means) / stds

    return pd.DataFrame(X_scaled, columns=params["columns"], index=X.index)
