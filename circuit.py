import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister
from utils import Individual
import numpy as np
from qiskit.quantum_info import Statevector
from qiskit_aer import StatevectorSimulator
from qiskit.visualization import plot_state_qsphere, plot_state_city, plot_state_hinton
from qiskit_aer import AerSimulator

def build_circuit(X: np.ndarray, i: Individual) -> QuantumCircuit:
    """
    Builds a single-qubit quantum circuit parameterized by the input array X and the individual's theta values.

    Args:
        X (np.ndarray): Input array of features, expected to have at least two elements.
        i (Individual): An object containing a 'theta' attribute, which is a list or array of parameters.

    Returns:
        QuantumCircuit: A quantum circuit with parameterized Ry and Rz rotations applied to the qubit.
    """
    circuit = QuantumCircuit(1)

    if len(i.theta) % 4 != 0  :
        raise ValueError("The dimension of theta must be a multiple of 4.")

    for k in range(0, len(i.theta), 4):
        eq1 = i.theta[k] * X[0] + i.theta[k+1] * X[1]
        circuit.ry(eq1, 0)

        eq2 = i.theta[k+2] * X[0] + i.theta[k + 3] * X[1]
        circuit.rz(eq2, 0)

    return circuit

def measurement_function(X: np.ndarray, i: Individual, n_shots: int):
    """
    Measures the probabilities of all computational basis states from a quantum circuit.

    Args:
        X (np.ndarray): Input data used to build the quantum circuit.
        i (Individual): An individual used to parameterize the circuit.
        n_shots (int): Number of measurement shots to perform in the simulation.

    Returns:
        tuple[float, float]: The probabilities (prob_0, prob_1) of measuring states |0⟩ and |1⟩.
    """
    circuit = build_circuit(X, i)

    # Create a measurement circuit
    meas = QuantumCircuit(1, 1)
    meas.measure(0, 0)

    qc = meas.compose(circuit, [0], front=True)

    aer_sim = AerSimulator()
    job_sim = aer_sim.run(qc, shots=n_shots)

    result_sim = job_sim.result()
    counts = result_sim.get_counts()

    # Calculate probabilities for both states
    total_shots = sum(counts.values())
    prob_0 = counts.get('0', 0) / total_shots
    prob_1 = counts.get('1', 0) / total_shots

    return prob_0, prob_1


