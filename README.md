# 🏠 House Price Prediction — v2 Enhanced ML Pipeline

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

## 📈 Results (Summary)

### Custom Gradient Descent Regressor
- Converged in ~1500 iterations  
- Test RMSE: ~75,000 USD  
- Test R²: ~0.58  

### Best overall model (from report):
- Ridge Regression with α = 1.0  
- Test RMSE: ~73,000 USD  
- Test R²: ~0.60  

Do NOT put full tables — those belong in the report. -> Ket takeaways (1-2 bullets)

8. Link to full report

Link to report.md:

For the full technical explanation, see report.md

This is where you point anyone who wants deep detail.

9. Future Work
in the next project combine hyperparameter tuning with cv 🔥
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
MIT License — feel free to use and modify with attribution.
See the [`LICENSE`](./LICENSE) file for full details.

---

## 👤 Author
**Ilian Khankhalaev**  
_BSc Computing Science, Simon Fraser University_  
📍 Vancouver, BC  |  [florykhan@gmail.com](mailto:florykhan@gmail.com)  |  [GitHub](https://github.com/florykhan)  |  [LinkedIn](https://www.linkedin.com/in/ilian-khankhalaev/)
