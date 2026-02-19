# 📘 House Price Prediction — Version 2
### *End-to-End Machine Learning Pipeline with Linear Models & Custom Gradient Descent*

---

## 1. Introduction

This report documents **Version 2** of a complete machine learning pipeline for predicting median house prices in California. The project moves beyond a single-model baseline by introducing **modular source code**, **engineered features**, **multiple linear models** (OLS, Ridge, Lasso), **systematic 5-fold cross-validation**, **hyperparameter tuning**, and a **custom Gradient Descent Regressor** implemented from scratch. Every step—from raw data to final metrics—is traceable through notebooks and reusable Python modules.

**Why it matters.** Accurate, interpretable price prediction supports buyers and sellers in decision-making and helps policymakers understand housing markets. Beyond the application, this pipeline demonstrates **production-oriented ML engineering**: clear separation of preprocessing, training, and evaluation; reproducible runs via fixed random seeds; and a single command (`python3 -m src.train`) to retrain and evaluate. The comparison between a hand-built optimizer and scikit-learn’s closed-form solution also illustrates the trade-offs between flexibility, control, and performance in linear regression.

---

## 2. Dataset

The pipeline uses the **California Housing** dataset, a well-known regression benchmark derived from the 1990 U.S. Census.

| Property | Description |
|----------|-------------|
| **Source** | `sklearn.datasets.fetch_california_housing` (or CSV in `data/raw/`) |
| **Samples** | ~20,640 block groups |
| **Target** | `median_house_value` (USD) |
| **Type** | Tabular, numerical regression |

**Predictors (8 original features):**

- **Location:** `latitude`, `longitude`
- **Housing structure:** `housing_median_age`, `total_rooms`, `total_bedrooms`, `households`
- **Demographics:** `median_income`, `population`
- **Categorical (preprocessed):** `ocean_proximity` (one-hot encoded in feature engineering)

Preprocessing and feature transformations are implemented in `src/feature_engineering.py`. Summary statistics, distributions, and correlations are explored in `notebooks/01_exploration.ipynb`; no processed datasets are persisted—all transformations are applied on the fly for reproducibility and to avoid storing large files in version control.

---

## 3. Project Structure

The repository is organized for clarity and reuse: notebooks drive the narrative and experiments, while `src/` holds the logic used by both the pipeline and the notebooks.

```
house-price-ml-v2/
│
├── notebooks/
│   ├── 01_exploration.ipynb          # EDA, distributions, correlations
│   ├── 02_model_evaluation.ipynb     # Custom Gradient Descent training & evaluation
│   ├── 03_sklearn_baseline.ipynb     # Sklearn LinearRegression baseline
│   ├── 04_cross_validation.ipynb     # 5-fold CV: OLS, Ridge, Lasso
│   └── 05_pipeline_demo.ipynb        # End-to-end pipeline demo
│
├── reports/
│   ├── metrics/                      # Exported JSON metrics (CV, test scores)
│   ├── plots/                        # Figures (loss curves, predicted vs actual, residuals, CV)
│   └── report.md                     # This document
│
├── src/
│   ├── config.py                     # Paths, random_state, learning_rate, n_splits, etc.
│   ├── data_loader.py                # Load raw data, train/test split
│   ├── evaluation.py                 # MAE, RMSE, R²
│   ├── feature_engineering.py        # Transformations and derived features
│   ├── gradient_descent.py           # Custom LinearRegressionGD
│   ├── hyperparameter_tuning.py      # Grid search over α (Ridge/Lasso)
│   ├── model_io.py                   # Save/load pipeline artifacts
│   ├── preprocessing.py              # Standardization (fit on train, apply to train/test)
│   └── train.py                      # Full pipeline entry point
│
├── README.md
└── requirements.txt
```

Version 2 emphasizes **separation of concerns**: data loading, feature engineering, training, and evaluation live in dedicated modules so that notebooks stay readable and the same code path can be run headless via `python3 -m src.train`.

---

## 4. Methodology

### 4.1 Feature Engineering

Feature engineering is centralized in `src/feature_engineering.py`. The pipeline applies the following transformations before model training:

- **Categorical encoding:** `ocean_proximity` is one-hot encoded (drop-first) so that all inputs are numeric.
- **Log transforms:** For skewed predictors, the pipeline adds \(\log(1 + x)\) versions of `median_income`, `total_rooms`, `total_bedrooms`, `population`, and `households` to reduce skew and improve linear fit.
- **Ratio features:** Safe division (avoiding division by zero) is used to create:
  - `rooms_per_household`
  - `bedrooms_per_room`
  - `population_per_household`
- **Polynomial term:** A squared term for `median_income` (`median_income_sq`) is included to capture a simple nonlinear effect of income on price.
- **Robustness:** Infinities are replaced and remaining NaNs are filled so that downstream scaling and model training never see missing or invalid values.

All transformations are **consistent** between training and inference: the same function is used in the pipeline and in the notebooks, and standardization parameters are fit only on the training set (see `src/preprocessing.py`).

---

### 4.2 Models Implemented

Four model types are used for comparison and education:

| Model | Implementation | Role |
|-------|----------------|------|
| **OLS (Linear Regression)** | scikit-learn `LinearRegression` | Closed-form baseline; best linear fit in least-squares sense. |
| **Ridge Regression** | scikit-learn | L2 penalty to shrink coefficients and improve stability, especially with correlated features. |
| **Lasso Regression** | scikit-learn | L1 penalty to encourage sparsity; useful for interpretation and feature selection. |
| **Gradient Descent Regressor** | Custom `LinearRegressionGD` in `src/gradient_descent.py` | Iterative optimization with configurable learning rate, iteration count, and optional L1/L2; used to illustrate optimization and compare to OLS. |

The custom Gradient Descent model minimizes **MSE** (with optional L1/L2 terms). It supports:

- Configurable **learning rate** and **number of iterations** (defaults in `src/config.py`: e.g. 0.05 and 5000).
- **Convergence tracking** via a stored loss history, which is plotted in the notebooks and in the figures referenced below.
- **Batch updates:** gradients are computed over the full training set each iteration.

---

### 4.3 Training Pipeline

The end-to-end workflow is implemented in `src/train.py` and can be run as:

```bash
python3 -m src.train
```

Steps performed:

1. **Load config** (paths, `random_state`, test size, learning rate, iterations, etc.).
2. **Load raw data** from `data/raw/` (e.g. `housing.csv`).
3. **Apply feature engineering** (same transformations as in the notebooks).
4. **Train/test split** (e.g. 80/20, with `random_state=42` for reproducibility).
5. **Standardization:** compute mean and standard deviation on the training set only; scale both train and test with these parameters to avoid leakage.
6. **Train** the Gradient Descent model (or the model configured in the pipeline).
7. **Evaluate** on the test set (MAE, RMSE, R²).
8. **Save** model artifacts (e.g. to `models/`) and optionally run hyperparameter tuning.

This design encapsulates the full workflow in one entry point and ensures that preprocessing and evaluation are consistent with the notebook-based experiments.

---

## 5. Cross-Validation

To assess **stability and variance** of the linear models, 5-fold cross-validation is used in `04_cross_validation.ipynb`. Each model (OLS, Ridge, Lasso) is trained and evaluated on five different train/validation splits; the same splits are used for all models so that comparisons are fair.

**Reported quantities:**

- **Mean R²** across folds — central tendency of performance.
- **Standard deviation of R²** — variability across folds; lower std suggests more stable performance.
- **Fold-level R²** — per-fold scores for debugging and to spot difficult splits.

> **Note:**  
> In production, hyperparameter tuning is often done *inside* cross-validation (e.g. nested CV or `GridSearchCV`). Here, CV and tuning are kept separate for clarity and modularity; the notebooks and `src/hyperparameter_tuning.py` can be combined in a later iteration (e.g. Future Work).

---

## 6. Hyperparameter Tuning

Ridge and Lasso depend on a regularization strength **α**. The project uses a **grid search** over a predefined set of α values, implemented in `src/hyperparameter_tuning.py` and demonstrated in the notebooks. For each α:

- Models are trained on the training (or train part of a split) set.
- Validation performance (e.g. R²) is computed.
- The α with the best validation R² is selected.

This process is kept separate from the 5-fold CV used for the main results table, so that the reported CV metrics reflect a fixed configuration rather than tuning on the same folds. Reproducing the exact best α and tuning curves is done by re-running the corresponding notebook cells.

---

## 7. Results

### 7.1 Cross-Validation Results

Below are the **aggregated 5-fold CV** results (mean and standard deviation of R²) and the **per-fold R²** for each model. All use the same data splits and feature pipeline.

**Summary (mean ± std R²):**

| Model | Mean R² | Std R² |
|-------|---------|--------|
| OLS | 0.5738 | 0.2087 |
| Ridge | **0.5842** | **0.1879** |
| Lasso | 0.5740 | 0.2082 |

**Per-fold R²:**

| Fold | OLS | Ridge | Lasso |
|------|-----|-------|-------|
| 1 | 0.1569 | 0.2089 | 0.1581 |
| 2 | 0.6922 | 0.6921 | 0.6922 |
| 3 | 0.6866 | 0.6866 | 0.6866 |
| 4 | 0.6623 | 0.6624 | 0.6623 |
| 5 | 0.6709 | 0.6708 | 0.6709 |

**Interpretation:** Ridge achieves the **highest mean R²** (0.5842) and the **lowest standard deviation** (0.1879), so it is both slightly more accurate on average and more **stable** across folds. Fold 1 is notably harder (R² ≈ 0.16) for all models—likely a geographic or demographic subset where linear relationships are weaker. OLS and Lasso are very close in both mean and variance; the small gain from Ridge suggests mild benefit from L2 regularization under this setup.

**Visualizations:**

The following figures (generated by the notebooks and saved under `reports/plots/`) illustrate the distribution of scores across folds and the comparison between models.

- **R² distribution across folds:**  
  ![CV scores distribution](plots/04_cv_scores_distribution.png)

- **Model comparison (e.g. box or bar plot):**  
  ![CV comparison](plots/04_cv_comparison.png)

---

### 7.2 Final Test Performance

Single-run **test-set** metrics (no cross-validation on the test set) from the notebooks provide a direct comparison between the **custom Gradient Descent** regressor and the **sklearn Linear Regression (OLS)** baseline. Both use the same feature pipeline and the same train/test split for fairness.

**Custom Gradient Descent Regressor** (`02_model_evaluation.ipynb`):

| Metric | Value |
|--------|--------|
| MAE | 53,437.27 USD |
| RMSE | 74,661.01 USD |
| R² | 0.5746 |

**Sklearn Linear Regression (OLS)** (`03_sklearn_baseline.ipynb`):

| Metric | Value |
|--------|--------|
| MAE | 48,572.07 USD |
| RMSE | 66,645.49 USD |
| R² | 0.6752 |

**Interpretation:** The sklearn OLS baseline **outperforms** the custom Gradient Descent on this split: higher R² (0.6752 vs 0.5746) and lower MAE and RMSE. This is expected: with the same features and no regularization, the closed-form least-squares solution is optimal for MSE, while the custom GD implementation may stop at a slightly different point (e.g. finite iterations, learning rate). The custom model still explains a substantial share of variance (~57%) and serves as a clear **educational baseline** and a reference for understanding iterative optimization vs. closed-form solutions.

**Predicted vs actual and residuals:**

The next four figures show how predictions line up with actual values and how errors are distributed. Tighter scatter around the diagonal in predicted-vs-actual plots and residuals centered at zero with relatively symmetric tails indicate better fit.

- **Custom GD — predicted vs actual:**  
  ![Predicted vs actual (custom)](plots/02_predicted_actual_custom.png)

- **Custom GD — residual distribution:**  
  ![Residual distribution (custom)](plots/02_residual_distribution_custom.png)

- **Sklearn OLS — predicted vs actual:**  
  ![Predicted vs actual (sklearn)](plots/03_predicted_actual_sklearn.png)

- **Sklearn OLS — residual distribution:**  
  ![Residual distribution (sklearn)](plots/03_residual_distribution_sklearn.png)

---

## 8. Custom Gradient Descent Regressor

The **custom Gradient Descent** regressor is the centerpiece of the “from scratch” part of the project. Notebook `02_model_evaluation.ipynb` covers:

- **Update rules:** How the weights and bias are updated using the gradient of the MSE (and optional L1/L2 terms).
- **Cost function:** Definition of the loss and its role in monitoring convergence.
- **Convergence:** Plotting the loss over iterations to confirm that the objective decreases and stabilizes.
- **Comparison with OLS:** Side-by-side metrics and plots against sklearn’s `LinearRegression` to highlight the difference between iterative and closed-form solutions.

**Training loss curve:**

The plot below shows the **training loss (MSE)** over iterations. A smoothly decreasing curve that flattens out indicates stable convergence; the exact number of iterations and learning rate are set in `src/config.py` (e.g. 5000 iterations, learning rate 0.05).

![Training loss curve](plots/02_training_loss_curve.png)

This model demonstrates **optimization fundamentals** and the practical trade-offs of implementing linear regression by hand versus using a library’s closed-form solver.

---

## 9. Limitations

- **Model class:** Only linear (and lightly regularized) models are used; no trees, kernels, or neural networks.
- **Hyperparameter search:** The grid over α is limited; a broader or finer grid, or nested CV, could yield different “best” settings.
- **Deployment:** The pipeline is built for experimentation and reproducibility, not for low-latency serving or production monitoring.
- **Data scope:** The California Housing dataset is domain-specific; conclusions may not transfer to other regions or time periods.
- **Single split for test metrics:** Final test MAE/RMSE/R² are from one train/test split; reporting confidence intervals would require multiple splits or a dedicated holdout strategy.

---

## 10. Future Work

- **ElasticNet:** Combine L1 and L2 in a single model for a richer regularization baseline.
- **Nested cross-validation:** Integrate hyperparameter tuning inside CV to avoid selection bias and report more reliable performance estimates.
- **Nonlinear models:** Add Random Forest, Gradient Boosting (e.g. XGBoost), or kernel methods to capture interactions and nonlinearities.
- **Features:** Explore more polynomial or interaction terms, or feature selection (e.g. via Lasso coefficients).
- **Interface:** Add a simple UI or API (e.g. FastAPI) for prediction demos and integration tests.
- **Experiment tracking:** Use MLflow or similar to log metrics, parameters, and artifacts across runs.

---

## 11. Reproducibility

The workflow is designed to be **fully reproducible**:

- **Code:** All logic lives in `src/` and is invoked from notebooks or `python3 -m src.train`.
- **Data:** Raw data path and split (e.g. 80/20) are fixed in `src/config.py`; no processed data is stored in the repo.
- **Randomness:** A fixed `random_state` (e.g. 42) is used for train/test split and any stochastic steps.
- **Artifacts:** Metrics are exported to `reports/metrics/` (JSON) and figures to `reports/plots/` so that runs can be compared and the report can reference exact numbers and images.

Re-running the notebooks in order and executing `python3 -m src.train` after placing the raw dataset in `data/raw/` should regenerate the metrics and plots referenced in this document.

---

*End of Report*

