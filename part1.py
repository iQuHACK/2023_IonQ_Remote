import qiskit
from qiskit import Aer
import qiskit.circuit.library
from qiskit import transpile, assemble
from qiskit.visualization import plot_histogram

import numpy as np
import math

SIZE = 28
N = math.ceil(math.log2(SIZE))
NB_QUBITS = 3 #2*N + 1

def load_images(path: str):
    return np.load(path)

def pixel_value_to_theta(pixel: int) -> float:
    return pixel / 255 * (np.pi/2)

def theta_to_pixel_value(theta: float) -> int:
    return int(theta / (np.pi/2) * 255)

def switch_x(circuit: qiskit.QuantumCircuit, pixel_position: int):
    if pixel_position == 0:
        return
    
    previous_position = pixel_position - 1

    N = NB_QUBITS - 1
    prev_repr = np.binary_repr(previous_position, width=N)
    curr_repr = np.binary_repr(pixel_position, width=N)

    for i in range(N):
        if prev_repr[i] != curr_repr[i]:
            circuit.x(i)
    return circuit

def frqi_encode(image):
    circuit = qiskit.QuantumCircuit(NB_QUBITS)

    # Get the theta values for each pixel
    thetas = [pixel_value_to_theta(pixel) for pixel in image]

    # Apply Hadamard gates for all qubits except the last one
    for i in range(NB_QUBITS - 1):
        circuit.h(i)

    # Apply the rotation gates
    for i in range(SIZE * SIZE):
        theta = thetas[i]
        # Get the binary representation of the index
        binary = bin(i)[2:].zfill(N)
        print(binary)
        # Apply the rotation gates
        for j in range(N):
            if binary[j] == '1':
                circuit.x(j)
        c3ry = qiskit.circuit.library.RYGate(theta).control(NB_QUBITS - 1)
        circuit.append(c3ry, list(range(NB_QUBITS)))
        for j in range(N):
            if binary[j] == '1':
                circuit.x(j)
    
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