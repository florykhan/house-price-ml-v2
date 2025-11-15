from __future__ import annotations

import pandas as pd

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Placeholder for feature engineering.
    For now, just returns the dataframe untouched.

    Later we will:
    - add ratios (rooms per household, etc.)
    - log transforms
    - polynomial features
    """
    # TODO: implement real feature engineering
    return df.copy()
