import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister

def encoder(image):
    n_qubits = 16
    qc = QuantumCircuit(16)
    for i, pixel_value in enumerate(np.flatten(image)):
        if pixel_value:
            qc.x(i)
    return qc
