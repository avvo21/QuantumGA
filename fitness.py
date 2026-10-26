import numpy as np
from circuit import measurement_function
from utils import Individual
from joblib import Parallel, delayed

def cross_entropy_loss(individual: Individual, X: np.ndarray, y: np.ndarray, n_shots: int) -> float:

    # Parallel execution with joblib
    losses = Parallel(n_jobs=-1)( # Use all available cores
        delayed(compute_single_loss)(i, X, y, individual, n_shots) for i in range(len(X))
    )

    # Convert to list to handle joblib return type
    losses_list = list(losses)
    total_loss = sum(loss for loss in losses_list if loss is not None)
    return float(total_loss / len(X))


def compute_single_loss(i: int, X: np.ndarray, y: np.ndarray, individual: Individual, n_shots: int) -> float:
    """Helper function to compute loss for a single sample"""
    prob_0, prob_1 = measurement_function(X[i], individual, n_shots)

    # To avoid log(0)
    eps = 1e-10
    prob_0 = max(prob_0, eps)
    prob_1 = max(prob_1, eps)

    if y[i] == 0:
        return -np.log(prob_0)
    else:
        return -np.log(prob_1)