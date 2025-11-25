from __future__ import annotations

import pandas as pd
import numpy as np

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering module.

    Applies a series of transformations to enhance model performance, including:
    - ratio features (e.g., rooms per household, bedrooms per room)
    - log transformations for skewed numerical features
    - one-hot encoding for categorical variables
    - polynomial features
    - standard preprocessing utilities for ML pipelines
    """
    df = df.copy()

    # 0. One-Hot Encoding
    if "ocean_proximity" in df.columns:
        df = pd.get_dummies(df, columns=["ocean_proximity"], drop_first=True)
        
    # 1. Log transforms (avoid 0)
    log_features = [
        "median_income",
        "total_rooms",
        "total_bedrooms",
        "population",
        "households",
    ]

    for col in log_features:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col].fillna(0))   # log(1 + x)

    # 2. Add ratio/interaction features + safe division
    def safe_div(a, b):
        return (a / b.replace(0, np.nan)).fillna(0) # replace all zeros in 'b' with NaN -> since dividing by NaN produces NaN
    
    if "total_rooms" in df.columns and "households" in df.columns:
        df["rooms_per_household"] = safe_div(df["total_rooms"], df["households"])

    if "total_bedrooms" in df.columns and "total_rooms" in df.columns:
        df["bedrooms_per_room"] = safe_div(df["total_bedrooms"], df["total_rooms"])

    if "population" in df.columns and "households" in df.columns:
        df["population_per_household"] = safe_div(df["population"], df["households"])

    # 3. Optional low-degree polynomial
    # Only apply polynomial to median_income (most impactful)
    if "median_income" in df.columns:
        df["median_income_sq"] = df["median_income"].fillna(0) ** 2

    # 4. Fill any created NaNs
    df = df.replace([np.inf, -np.inf], 0)
    df = df.fillna(0)

    return df