from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # Paths
    project_root: Path = Path(__file__).resolve().parents[1]
    data_path: Path = project_root / "data" / "raw" / "housing.csv"  # <-- same CSV as v1
    processed_data_dir: Path = project_root / "data" / "processed"
    models_dir: Path = project_root / "models"
    reports_dir: Path = project_root / "reports"

    # Data split
    test_size: float = 0.2
    random_state: int = 42

    # Gradient Descent defaults
    learning_rate: float = 0.05
    n_iterations: int = 5000

    # Regularization
    use_l2: bool = False
    use_l1: bool = False
    lambda_: float = 0.0  # reg strength

    # Cross-validation
    n_splits: int = 5

    # Hyperparameter Tuning
    use_hyperparameter_tuning: bool = False

