import numpy as np
from typing import List, Tuple

from .gradient_descent import LinearRegressionGD
from .evaluation import evaluate_regression


def grid_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> Tuple[Tuple[float, int, float, float], float, List[tuple]]:

    # Medium grid search ranges
    learning_rates = [0.001, 0.005, 0.01, 0.05]
    iterations = [500, 1500, 3000, 5000]
    l2_values = [0, 0.001, 0.01, 0.1]
    l1_values = [0, 0.0005, 0.001]

    best_score = -np.inf
    best_params = None
    results = []

    total_runs = (
        len(learning_rates)
        * len(iterations)
        * len(l2_values)
        * len(l1_values)
    )
    print(f"Starting grid search over {total_runs} combinations...\n")

    run_idx = 1

    for lr in learning_rates:
        for iters in iterations:
            for l2 in l2_values:
                for l1 in l1_values:

                    print(f"[{run_idx}/{total_runs}] Testing lr={lr}, iters={iters}, L1={l1}, L2={l2}")

                    # Train model
                    model = LinearRegressionGD(
                        learning_rate=lr,
                        n_iterations=iters,
                        l1_lambda=l1,
                        l2_lambda=l2,
                    )

                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)

                    # Skip this parameter set if predictions are invalid
                    if np.isnan(preds).any() or np.isinf(preds).any():
                        print("Warning: invalid predictions (NaN/inf), skipping this parameter set")
                        run_idx += 1
                        continue


                    metrics = evaluate_regression(y_test, preds)
                    r2 = metrics["r2"]

                    results.append((lr, iters, l1, l2, r2))

                    if r2 > best_score:
                        best_score = r2
                        best_params = (lr, iters, l1, l2)

                    run_idx += 1

    print("\nGrid Search Complete.")
    print(f"Best R² score: {best_score:.5f}")
    print(f"Best params: learning_rate={best_params[0]}, "
          f"iterations={best_params[1]}, L1={best_params[2]}, L2={best_params[3]}")

    return best_params, best_score, results
