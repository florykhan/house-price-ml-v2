from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from .config import Config
from .data_loader import load_raw_data, train_test_split_data
from .feature_engineering import apply_feature_engineering
from .preprocessing import compute_standardization_params, apply_standardization
from .gradient_descent import LinearRegressionGD
from .evaluation import evaluate_regression
from .hyperparameter_tuning import grid_search
from .model_io import save_model


def main() -> None:
    """
    Full training pipeline:
    1. Load config
    2. Load raw data
    3. Apply feature engineering
    4. Split train/test
    5. Compute standardization params on train
    6. Scale train and test
    7. Train model (gradient descent)
    8. Evaluate
    9. Save model
    10. Hyperparameter tuning (optional)
    """
    # ------ 1. Load config ------
    config = Config()

    # ------ 2. Load RAW data ------
    df = load_raw_data(config)

    # ------ 3. Feature engineering ------
    df_fe = apply_feature_engineering(df)

    # ------ 4. Train/test split ------
    X_train, X_test, y_train, y_test = train_test_split_data(
        df_fe, target_col="median_house_value", config=config
    )

    # ------ 5. Compute standardization ------
    std_params = compute_standardization_params(X_train)

    # ------ 6. Scale train and test ------
    X_train_scaled = apply_standardization(X_train, std_params)
    X_test_scaled = apply_standardization(X_test, std_params)

    # ------ 7. Train model ------
    model = LinearRegressionGD(
        learning_rate=config.learning_rate,
        n_iterations=config.n_iterations,
        l1_lambda=config.l1_lambda if config.use_l1 else 0,
        l2_lambda=config.l2_lambda if config.use_l2 else 0,
    )

    model.fit(
        X_train_scaled.to_numpy(),
        y_train.to_numpy(),
    )

    # Predictions
    y_pred = model.predict(X_test_scaled.to_numpy())

    # ------ 8. Evaluate ------
    metrics = evaluate_regression(y_test.to_numpy(), y_pred)
    print(metrics)

    # Plot training loss
    plt.plot(model.loss_history)
    plt.xlabel("Iteration")
    plt.ylabel("Loss (MSE + Regularization)")
    plt.title("Training Loss Curve")
    plt.show()

    # ------ 9. Save model ------
    save_model(
        filepath="models/model_v2.npz",
        model=model,
        std_params=std_params,
        feature_names=X_train_scaled.columns.tolist(),
        config=config,
    )

    # ------ 10. Hyperparameter tuning (Optional) ------
    if config.use_hyperparameter_tuning:
        print("\nRunning hyperparameter tuning...\n")

        best_params, best_r2, all_results = grid_search(
            X_train_scaled.to_numpy(),
            y_train.to_numpy(),
            X_test_scaled.to_numpy(),
            y_test.to_numpy(),
        )

        print("\nBest hyperparameters found:")
        print(f"learning_rate={best_params[0]}")
        print(f"iterations={best_params[1]}")
        print(f"L1={best_params[2]}")
        print(f"L2={best_params[3]}")
        print(f"Best R²={best_r2:.5f}")


if __name__ == "__main__":
    main()
