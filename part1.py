import qiskit
from qiskit import Aer
import qiskit.circuit.library
from qiskit import transpile, assemble
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import RYGate

from matplotlib import pyplot as plt

import numpy as np
import math

SIZE = 3#28
N = math.ceil(math.log2(SIZE))
NB_QUBITS = 2*N + 1

def load_images(path: str):
    return np.load(path)

def pixel_value_to_theta(pixel: int) -> float:
    return pixel / 255 * (np.pi/2)

def theta_to_pixel_value(theta: float) -> int:
    return int(theta / (np.pi/2) * 255)

def switch_x(circuit: qiskit.QuantumCircuit, pixel_position: int):
    if pixel_position == 0:
        return circuit
    
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
    image = image.flatten()
    thetas = [pixel_value_to_theta(pixel) for pixel in image]

    # Apply Hadamard gates for all qubits except the last one
    for i in range(NB_QUBITS - 1):
        circuit.h(i)
    circuit.barrier()

    ry_qbits = list(range(NB_QUBITS))
    # Apply the rotation gates
    for i in range(SIZE * SIZE):
        theta = thetas[i]
        circuit = switch_x(circuit, i)
        print(i, theta)
        c3ry = RYGate(2*theta).control(NB_QUBITS - 1)
        circuit.append(c3ry, ry_qbits)

        #qc = circuit
        # circuit.cry(theta,0,2)
        # circuit.cx(0,1)
        # circuit.cry(-theta,1,2)
        # circuit.cx(0,1)
        # circuit.cry(theta,1,2)
        circuit.barrier()
    
    circuit.measure_all()
    # Print the circuit
    print(circuit)
    return circuit

def decode(histogram):
    nb_px = SIZE * SIZE
    img = np.zeros(nb_px)
    print(histogram)

    for i in range(nb_px):
        print(i)
        bin_str = np.binary_repr(i, width=NB_QUBITS - 1)
        print(bin_str)
        cos_str = "0" + bin_str[::-1]
        sin_str = "1" + bin_str[::-1]

        n0 = 2 ** NB_QUBITS

        if cos_str in histogram:
            prob_cos = histogram[cos_str] * n0
        else:
            prob_cos = 0

        # not needed?
        if sin_str in histogram:
            prob_sin = histogram[sin_str] * n0
        else:
            prob_sin = 0

        print(n0, cos_str, sin_str)
        prob_cos = np.clip(prob_cos, 0, 1)
        prob_sin = np.clip(prob_sin, 0, 1)

        print(prob_cos, prob_sin)
        theta = math.acos(prob_cos)
        print(theta)

        img[i] = theta_to_pixel_value(theta)

    return img.reshape(SIZE, SIZE)

def get_proba(counts):
    sums = sum(map(lambda x: x[1], counts.items()))
    return {key: value / sums for key, value in counts.items()}

if __name__ == "__main__":
    image = load_images('data/images.npy')[5]
    if image.max() != 0:
        image = image/image.max() * 255
    # plt.imshow(image, cmap='gray')
    # plt.show()
    #print(image.min(), image.max())

    #print(image)
    # print((image.flatten() * 255).astype(int))
    image = np.array([0, 255, 0, 255, 0, 255, 0, 255, 120, 0, 255, 0, 255, 0, 255, 0, 255, 120])
    image = image[:SIZE*SIZE]
    print(2**SIZE-1)
    circuit = frqi_encode(image)
    # Simulate the circuit
    aer_sim = Aer.get_backend('aer_simulator')
    t_qc = transpile(circuit, aer_sim)
    qobj = assemble(t_qc, shots=16384)
    result = aer_sim.run(qobj).result()
    counts = result.get_counts(circuit)
    print(counts)

    # Decode the histogram
    img = decode(get_proba(counts))
    print(img.flatten())  