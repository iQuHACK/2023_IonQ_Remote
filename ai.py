

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector


images=np.load('data/images.npy')
labels=np.load('data/labels.npy')

image = images[0]
def encode(image):
    # Pre-process the image
    normalized_image = (image / 255).flatten()

    # Map the image pixels to quantum amplitudes
    num_qubits = int(np.ceil(np.log2(len(normalized_image))))
    state = np.zeros(2**num_qubits)
    state[:len(normalized_image)] = np.sqrt(normalized_image)

    # Create a quantum circuit from the compressed quantum state
    qr = QuantumRegister(num_qubits)
    circuit = QuantumCircuit(qr)
    circuit.initialize(state, qr)

    return circuit
