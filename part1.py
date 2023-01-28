import qiskit
from qiskit import Aer
import qiskit.circuit.library
from qiskit import transpile, assemble
from qiskit.visualization import plot_histogram

import numpy as np
import math

SIZE = 28
N = math.ceil(math.log2(SIZE))
NB_QUBITS = 3 # 2*N + 1

def load_images(path: str):
    return np.load(path)

def pixel_value_to_theta(pixel: int) -> float:
    return pixel / 255 * (np.pi/2)

def theta_to_pixel_value(theta: float) -> int:
    return int(theta / (np.pi/2) * 255)

def frqi_encode(image):
    circuit = qiskit.QuantumCircuit(NB_QUBITS)

    # Get the theta values for each pixel
    thetas = [pixel_value_to_theta(pixel) for pixel in image]

    # Apply Hadamard gates for all qubits except the last one
    for i in range(NB_QUBITS - 1):
        circuit.h(i)
    
    # Apply the rotation gates
    theta = 0
    c3ry = qiskit.circuit.library.RYGate(theta).control(2)
    circuit.append(c3ry, [0, 1, 2])

    circuit.x(1)
    c3ry = qiskit.circuit.library.RYGate(theta).control(2)
    circuit.append(c3ry, [0, 1, 2])

    circuit.x(0)
    circuit.x(1)
    c3ry = qiskit.circuit.library.RYGate(theta).control(2)
    circuit.append(c3ry, [0, 1, 2])

    circuit.x(1)
    c3ry = qiskit.circuit.library.RYGate(theta).control(2)
    circuit.append(c3ry, [0, 1, 2])
    
    circuit.measure_all()
    # Print the circuit
    print(circuit)
    return circuit

if __name__ == "__main__":
    image = load_images('data/images.npy')[0]
    circuit = frqi_encode(image)
    # Simulate the circuit
    aer_sim = Aer.get_backend('aer_simulator')
    t_qc = transpile(circuit, aer_sim)
    qobj = assemble(t_qc, shots=4096)
    result = aer_sim.run(qobj).result()
    counts = result.get_counts(circuit)
    print(counts)