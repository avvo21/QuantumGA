import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from typing import Tuple

def generate_circle_dataset(n_samples: int = 250, radius: float = 0.5, noise: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
    X = np.random.uniform(-1, 1, (n_samples, 2))
    distances = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    y = (distances <= radius).astype(int)
    X += np.random.normal(0, noise, X.shape)
    return X, y

def train_test_split( X: np.ndarray, y: np.ndarray, train_ratio: float = 0.8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = X.shape[0]
    n_train = int(n_samples * train_ratio)
    indices = np.random.permutation(n_samples)
    train_idx, test_idx = indices[:n_train], indices[n_train:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    return X_train, y_train, X_test, y_test

def plot_circle_dataset(X: np.ndarray, y: np.ndarray, radius: float = 0.5) -> None:
    fig, ax = plt.subplots()
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', edgecolor='k', alpha=0.7, label='Points')
    circle = Circle((0, 0), radius, color='b', fill=False, linewidth=2, label='Circle Border')
    ax.add_patch(circle)
    ax.set_aspect('equal')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f'Dataset - {X.shape[0]} points')
    plt.show()

def normalize_X_to_pi(X: np.ndarray) -> np.ndarray:
    """Normalize X from any range to [0, π]"""

    e = 1e-8

    # First normalize to [0, 1]
    X_min = X.min()
    X_max = X.max()
    X_normalized = (X - X_min) / (X_max - X_min)
    
    # Then scale to [0, 2π)
    X_norm = X_normalized * (2*np.pi - e)
    return X_norm