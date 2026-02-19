# 🏠 House Price Prediction | v2 Enhanced ML Pipeline

This repository implements an **advanced, custom modular machine learning pipeline** for predicting median house prices in California. It represents the **second version (v2)** of the project, introducing engineered features, regularized linear models, cross-validation, hyperparameter tuning, and a custom Gradient Descent implementation built from scratch — all organized into reusable **source modules** powering the end-to-end training pipeline.

---

## 🎯 Project Overview

The goal of this version is to:
- Build a **fully modular, extensible machine learning pipeline** for structured tabular data.
- Introduce **feature engineering and standardization** to improve model stability and performance.
- Implement and compare **multiple linear models**, including OLS, Ridge, Lasso, and a custom Gradient Descent Regressor built from scratch.
- Evaluate model robustness through **5-fold cross-validation** and targeted hyperparameter tuning.
- Establish a **reproducible training workflow** that cleanly separates preprocessing, model training, evaluation, and inference.

This version focuses on **engineering best practices**, enabling experimentation, modularity, and reliable performance benchmarking across models.

---

## 🔄 What’s New in v2 (Compared to v1)

A quick overview of improvements introduced in Version 2:

| Aspect | v1 Baseline | v2 Enhanced Pipeline |
|--------|-------------|----------------------|
| Feature Engineering | Minimal | Full transformations + scaling |
| Models | OLS only | GD (custom), OLS, Ridge, Lasso |
| Cross-Validation | No | Yes — 5-fold |
| Pipeline | Notebook-only | Modular Python pipeline (`src/`) |
| Performance | Higher RMSE | Lower RMSE after FE + regularization |

---

## ✨ Key Features

- **Custom Modular ML pipeline** (`src/` folder) for clean separation of preprocessing, training, evaluation, and utilities.
- **Feature engineering & standardization**, ensuring consistent transformations across training and inference.
- **Multiple linear models**:
  - OLS (Linear Regression)
  - Ridge Regression (L2)
  - Lasso Regression (L1)
- **Custom Gradient Descent Regressor** implemented from scratch with configurable learning rate, iterations, and convergence tracking.
- **5-fold Cross-Validation** for assessing model stability and variance.
- **Hyperparameter tuning** for regularized models using a simple grid search workflow.
- **Reproducible training pipeline** (`python3 -m src.train`) for end-to-end training and evaluation.
- **Five structured Jupyter notebooks** documenting the full development process.

---

## 🧱 Repository Structure
```
house-price-ml-v2/
│
├── data/                                  # Local dataset storage (not included in Git intentionally for privacy / size)
│   ├── processed/                         # Cleaned / transformed data (not saved in this project; kept ephemeral)
│   └── raw/                               # Unmodified input data (as downloaded)
│   
├── models/                                # Saved model artifacts (not included in Git to avoid large files and ensure reproducible training)
│
├── notebooks/                             # Full development workflow (v2 notebooks)
│   ├── 01_exploration.ipynb               # Initial EDA, data inspection, distributions, correlations
│   ├── 02_model_evaluation.ipynb          # Training and evaluating the custom Gradient Descent Regressor
│   ├── 03_sklearn_baseline.ipynb          # Baseline Linear Regression using scikit-learn (LinearRegression)
│   ├── 04_cross_validation.ipynb          # 5-fold CV comparing OLS, Ridge, and Lasso models, stability analysis
│   └── 05_pipeline_demo.ipynb             # End-to-end demonstration of the modular training pipeline
│
├── reports/                               # Project documentation and reports
│   ├── metrics/
│   ├── plots/
│   └── report.md                          # Detailed technical write-up for Version 2
│
├── src/                                   # Modular machine learning pipeline (from scratch)
│   ├── __init__.py                        # Marks directory as a Python package
│   ├── config.py                          # Centralized configuration / constants
│   ├── data_loader.py                     # Data loading utilities for datasets
│   ├── evaluation.py                      # Metrics, scoring utilities, model evaluation logic
│   ├── feature_engineering.py             # Data cleaning, feature transformations, scaling logic
│   ├── gradient_descent.py                # Custom Gradient Descent Regressor
│   ├── hyperparameter_tuning.py           # Grid search utilities
│   ├── model_io.py                        # Save/load functions for pipeline components and artifacts
│   ├── preprocessing.py                   # Preprocessing utilities (standardization)
│   └── train.py                           # End-to-end pipeline (run via: python3 -m src.train)
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

> 🗒️ **Note:**  
> Version 2 uses a **fully modular architecture** inside `src/`, and the `notebooks/` directory follows a clean, sequential workflow from exploration → baselines → cross-validation → final pipeline.
> - The `data/` directory is **not tracked by Git** to avoid storing large files and to keep the repository lightweight. Only *raw* data should be placed here. The `data/processed/` folder exists as a placeholder for a scalable workflow, but **no processed data is saved in this project** — all transformations are generated dynamically by the pipeline / notebooks.  
> - The `models/` directory is also **excluded from Git**, since model artifacts are generated during training and can be reproduced at any time by running the pipeline.

---

## 🧰 Run Locally

You can run this project on your machine using **Python 3.11** and `venv`.

### 1️⃣ Clone the repository
**HTTPS (recommended for most users):**
```bash
git clone https://github.com/florykhan/house-price-ml-v2.git
cd house-price-ml-v2
```
**SSH (for users who have SSH keys configured):**
```bash
git clone git@github.com:florykhan/house-price-ml-v2.git
cd house-price-ml-v2
```

### 2️⃣ Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows
```
### 3️⃣ Install dependencies
```bash
pip3 install -r requirements.txt
```

### 4️⃣ Add the dataset
Place `housing.csv` inside the `data/raw/` folder:
```bash
house-price-ml-v2/data/raw/housing.csv
```
> 📥 **Download the dataset:**  
> - Via scikit-learn:  
>   ```python
>   from sklearn.datasets import fetch_california_housing
>   data = fetch_california_housing(as_frame=True)
>   ```
> - Or download the CSV from [Kaggle](https://www.kaggle.com/datasets/camnugent/california-housing-prices)
>
> ⚠️ **Note:**  Processed data is not saved in this project. All transformations are applied dynamically through the pipeline.

### 5️⃣ Run the training pipeline
This step is essential — the training pipeline performs all preprocessing, feature engineering, and model training needed for the notebooks to run correctly.

```bash
python3 -m src.train
```

### 6️⃣ Run the notebooks
Launch Jupyter and open the notebooks:
```bash
jupyter notebook
```

Recommended order:
- `01_exploration.ipynb` — data exploration
- `02_model_evaluation.ipynb` — custom Gradient Descent
- `03_sklearn_baseline.ipynb` — sklearn LinearRegression
- `04_cross_validation.ipynb` — OLS/Ridge/Lasso CV comparison
- `05_pipeline_demo.ipynb` — full end-to-end pipeline demo

---

## 📊 Results (Summary)

| **Custom Gradient Descent Regressor** | **Interpretation** |
|---------------------------------------|---------------------|
| • Converged in ~1500 iterations<br>• Test RMSE: **~74.6K USD**<br>• Test R²: **~0.57** | • Explains **~57%** of variance in housing prices.<br>• Captures strong **linear trends** (e.g., median income → price).<br>• Misses **nonlinear** and **interaction** effects → room for improvement. |



➡️ For full model comparisons (Ridge, Lasso, CV results, etc.), see the full report: [`reports/report.md`](reports/report.md)

---

## 📄 Full Technical Report

For the complete technical write-up — including model comparisons, cross-validation results, feature engineering details, and a full discussion of Version 2 improvements — see: [`reports/report.md`](reports/report.md). This document contains all deep-level explanations intended for reviewers who want to understand the full methodology behind the pipeline, models, experiments, and results.

---

## 🚀 Future Directions (Beyond This Project)

Although Version 2 concludes the development of this repository, there are several **advanced directions** worth exploring in future machine learning projects:

- **Combine hyperparameter tuning with cross-validation:** use integrated approaches like `GridSearchCV` or `RandomizedSearchCV` to avoid data leakage and optimize models more systematically.

- **Explore nonlinear models:** tree-based models (Random Forest, XGBoost), and kernel methods can capture complex relationships missed by linear models.

- **Build more modular and reusable ML tooling:** expand this project’s structure into a general-purpose ML pipeline framework usable across multiple datasets.

- **Add experiment tracking:** use MLflow or Weights & Biases to log metrics, save artifacts, compare runs, and manage the training lifecycle.

- **Improve production readiness:** add model APIs (FastAPI/Flask), Docker containers, simple CI/CD workflows, or model versioning for deployment-focused practice.

- **Work with more complex datasets:** move into NLP, image processing, or time-series problems to broaden your portfolio and ML expertise.

- **Add CI/CD workflows:** introduce lightweight GitHub Actions to automate code quality checks, formatting, dependency validation, and simple import tests, improving project reliability and demonstrating real-world ML engineering practices.

---

## 🧠 Tech Stack

- **Language:** Python 3.11  
- **Core Libraries:**  
  - `pandas`, `numpy`, `matplotlib`  
  - `scikit-learn`
- **Custom Components:**  
  - Custom Gradient Descent Regressor  
  - Reusable feature engineering, evaluation, and tuning modules (`src/` folder)
  - End-to-end training pipeline (`python3 -m src.train`)
- **Environment:** Jupyter Notebook / VS Code
- **Version Control:** Git + GitHub (SSH configured)

---

## 🧾 License
MIT License, feel free to use and modify with attribution. See the [`LICENSE`](./LICENSE) file for full details.

---

## 👤 Author
**Ilian Khankhalaev**  
_BSc Computing Science, Simon Fraser University_  
📍 Vancouver, BC  |  [florykhan@gmail.com](mailto:florykhan@gmail.com)  |  [GitHub](https://github.com/florykhan)  |  [LinkedIn](https://www.linkedin.com/in/ilian-khankhalaev/)
