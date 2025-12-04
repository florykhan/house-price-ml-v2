# 🏠 House Price Prediction — v2 Enhanced ML Pipeline

This repository implements an **advanced, modular machine learning pipeline** for predicting median house prices in California.  
It represents the **second version (v2)** of the project, introducing engineered features, regularized linear models, cross-validation, hyperparameter tuning, and a custom Gradient Descent implementation built from scratch.

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

## ✨ Key Features

- **Modular ML pipeline** (`src/` folder) for clean separation of preprocessing, training, evaluation, and utilities.
- **Feature engineering & standardization**, ensuring consistent transformations across training and inference.
- **Multiple linear models**:
  - OLS (Linear Regression)
  - Ridge Regression (L2)
  - Lasso Regression (L1)
- **Custom Gradient Descent Regressor** implemented from scratch with configurable learning rate, iterations, and convergence tracking.
- **5-fold Cross-Validation** for assessing model stability and variance.
- **Hyperparameter tuning** for regularized models using a simple grid search workflow.
- **Reproducible training pipeline** (`python -m src.training_pipeline`) for end-to-end training and evaluation.
- **Five structured Jupyter notebooks** documenting the full development process:
  - Feature engineering  
  - Model experiments  
  - Gradient Descent implementation  
  - Cross-validation  
  - End-to-end demonstration

---

## 🧱 Repository Structure
```
house-price-ml-v2/
│
├── data/                                  # Placeholder for dataset files (empty by default)
│
├── models/                                # Saved model artifacts (optional, can be ignored in Git)
│
├── notebooks/                             # Full development workflow (v2 notebooks)
│   ├── 01_exploration.ipynb               # Initial EDA, data inspection, distributions, correlations
│   ├── 02_model_evaluation.ipynb          # Baseline experiments, preprocessing tests, metric checks
│   ├── 03_sklearn_baseline.ipynb          # OLS, Ridge, Lasso training using scikit-learn
│   ├── 04_cross_validation.ipynb          # 5-fold CV for all models, stability analysis
│   └── 05_pipeline_demo.ipynb             # Full training pipeline demonstration (end-to-end)
│
├── reports/                               # Project documentation and reports
│   └── report.md                          # Detailed technical write-up for Version 2
│
├── src/                                   # Modular machine learning pipeline
│   ├── feature_engineering.py             # Data cleaning, transformations, standardization
│   ├── gradient_descent.py                # Custom Gradient Descent Regressor (from scratch)
│   ├── evaluation.py                      # Metrics, scoring utilities, model evaluation logic
│   ├── hyperparameter_tuning.py           # Grid search utilities for Ridge/Lasso
│   ├── training_pipeline.py               # End-to-end pipeline (run via: python -m src.training_pipeline)
│   └── utils.py                           # Shared helper functions (loading, saving, validation)
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

> 🗒️ **Note:**  
> Version 2 uses a **fully modular architecture** inside `src/`, and the `notebooks/` directory follows a clean, sequential workflow from exploration → baselines → cross-validation → final pipeline.

---

5. Installation

Just the commands:

pip install -r requirements.txt

6. How to Run

Example:

Run the full training pipeline 
python3 -m src.train
OR
python -m src.training_pipeline

Open the notebooks

01 — Feature Engineering

02 — Experimental Models

03 — Gradient Descent

04 — Cross-Validation

05 — Training Demo

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

10. Tech Stack

Python

scikit-learn

NumPy

Matplotlib / Seaborn

Jupyter

## 🧠 Tech Stack

- **Language:** Python 3.11  
- **Core Libraries:**  
  - `pandas`, `numpy`, `matplotlib`  
  - `scikit-learn`  
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
