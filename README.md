# Quantum Genetic Algorithm (QuantumGA)

A lightweight implementation of the hybrid quantum-classical classifier described in:

> **[Practicality of training a quantum-classical machine in the NISQ era](https://arxiv.org/abs/2401.12089)**  
> *Tarun Dutta, Alex Jin, Clarence Liu Huihong, José Ignacio Latorre, Manas Mukherjee* (arXiv:2401.12089, included as [`2401.12089v2.pdf`](2401.12089v2.pdf)).

---

## 📌 Summary

In NISQ quantum hardware, gradient-based optimizers (e.g., SGD, BFGS) often fail due to quantum measurement noise, hardware drift, and barren plateaus. This project implements a **derivative-free Genetic Algorithm (GA)** to train a variational quantum classifier, treating the quantum processor as a black-box evaluation oracle.

---

## ⚙️ How It Works

1. **Data Re-uploading Circuit (`circuit.py`)**  
   A single-qubit quantum classifier based on **Ansatz-2C** from the paper. The circuit applies repeated layers of non-commuting rotations combining input features $\vec{x} = [x_0, x_1]$ and variational parameters $\vec{\theta}$:
   $$U_l(\vec{\theta}, \vec{x}) = R_z(\theta_{4l+2}x_0 + \theta_{4l+3}x_1) \; R_y(\theta_{4l}x_0 + \theta_{4l+1}x_1)$$
   The state is measured in the $\{|0\rangle, |1\rangle\}$ basis with Qiskit Aer to determine class probabilities.

2. **Dataset & Objective (`dataset.py`, `fitness.py`)**  
   - **Task**: Binary classification of 2D points with a non-linear circular boundary ($x_0^2 + x_1^2 \le r^2$).
   - **Fitness**: Binary Cross-Entropy loss computed across the training set, accelerated using multi-core parallel execution (`joblib`).

3. **Genetic Algorithm Optimizer (`genetic.py`)**  
   Iteratively evolves candidate parameter vectors $\vec{\theta}$:
   - **Selection**: Rank-based selection (linear or exponential probabilities).
   - **Crossover**: Scattered crossover using a balanced binary mask.
   - **Mutation**: Gaussian or uniform random mutations with an exponentially decaying mutation rate ($p_{\text{mut}} \cdot \text{decay}^t$) to balance exploration and exploitation.
   - **Replacement**: Steady-State Selection (merging parents and offspring) or generational replacement with elitism.

---

## 📁 Project Structure

- [`circuit.py`](circuit.py) — Parameterized quantum circuit builder and measurement logic.
- [`genetic.py`](genetic.py) — Genetic algorithm engine (selection, crossover, mutation, evolution loop).
- [`fitness.py`](fitness.py) — Parallel cross-entropy loss computation.
- [`dataset.py`](dataset.py) — Circular dataset generation and plotting utilities.
- [`config.yaml`](config.yaml) — Hyperparameters for GA, circuit dimensions, and dataset.
- [`utils.py`](utils.py) — Config loader and `Individual` class definition.
- [`main.py`](main.py) — Entry point to train the model and display progress plots.

---

## 🚀 Quick Start

### 1. Requirements

```bash
pip install numpy matplotlib pyyaml joblib qiskit qiskit-aer
```

### 2. Run Training

```bash
python main.py
```

The script will:
- Generate and plot the circular dataset.
- Run the GA training loop and log generation loss & accuracy.
- Display fitness/accuracy convergence curves.
- Report train accuracy, test accuracy, and the generalization gap.

---

## 🔧 Configuration

All settings (population size, generations, mutation rate/type, selection method, circuit layers, and dataset size) can be tuned in [`config.yaml`](config.yaml).
