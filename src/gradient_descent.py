from __future__ import annotations

from typing import Optional

import numpy as np


class LinearRegressionGD:
    """
    Linear Regression model trained using batch gradient descent.

    This class provides:
    - fit(): parameter optimization via gradient descent
    - predict(): generate predictions for new input data
    - loss tracking across iterations
    - optional L1 and L2 regularization support
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_iterations: int = 1000,
        l1_lambda: float = 0.0,
        l2_lambda: float = 0.0,
    ) -> None:
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda

        self.weights: Optional[np.ndarray] = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        """
        Train the model using gradient descent.
        """
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):

            # Forward pass
            y_pred = X @ self.weights + self.bias

            # Compute residuals
            residuals = y_pred - y

            # MSE loss = J (cost function)
            loss = (1 /(2 * n_samples)) * np.sum(residuals ** 2)

            # Add L2 regularization
            if self.l2_lambda > 0:
                loss += (self.l2_lambda / (2 * n_samples)) * np.sum(self.weights ** 2)

            # Add L1 regularization
            if self.l1_lambda > 0:
                loss += (self.l1_lambda / n_samples) * np.sum(np.abs(self.weights))

            self.loss_history.append(float(loss))

            # Compute gradients
            dw = (1 / n_samples) * (X.T @ residuals)
            db = (1 / n_samples) * np.sum(residuals)

            # Add L2 gradient
            if self.l2_lambda > 0:
                dw += (self.l2_lambda / n_samples) * self.weights

            # Add L1 gradient (subgradient)
            if self.l1_lambda > 0:
                dw += (self.l1_lambda / n_samples) * np.sign(self.weights)

            # Parameter updates
            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

        return self



    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using learned weights.
        """
        if self.weights is None:
            raise ValueError("Model is not fitted yet.")
        return X @ self.weights + self.bias
