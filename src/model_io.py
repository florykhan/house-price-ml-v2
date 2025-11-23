import numpy as np


def save_model(filepath, model, std_params, feature_names, config):
    """
    Save model weights, bias, scaling params, feature names, and hyperparams.
    """
    np.savez(
        filepath,
        weights=model.weights,
        bias=model.bias,
        mean=std_params["mean"],
        std=std_params["std"],
        feature_names=np.array(feature_names),
        learning_rate=config.learning_rate,
        n_iterations=config.n_iterations,
        l1_lambda=model.l1_lambda,
        l2_lambda=model.l2_lambda,
    )
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load saved model components from .npz file.
    Returns a dict with all parts needed to reconstruct the model.
    """
    data = np.load(filepath, allow_pickle=True)

    return {
        "weights": data["weights"],
        "bias": data["bias"],
        "mean": data["mean"],
        "std": data["std"],
        "feature_names": data["feature_names"],
        "learning_rate": data["learning_rate"],
        "n_iterations": data["n_iterations"],
        "l1_lambda": data["l1_lambda"],
        "l2_lambda": data["l2_lambda"],
    }
