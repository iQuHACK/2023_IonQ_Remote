from collections import Counter
import qiskit
from qiskit import Aer
import qiskit.circuit.library
from qiskit import transpile, assemble
from qiskit.visualization import plot_histogram
from qiskit.circuit.library import RYGate

from matplotlib import pyplot as plt

import numpy as np
import math
from pprint import pprint

from typing import Dict, Union

from sklearn.metrics import mean_squared_error


# Image properties
SIZE = 5  # 28 # Image width
NB_PX_IMG = SIZE ** 2

# quantum parameters
N = math.ceil(math.log2(SIZE))
NB_QUBITS = 2 * N + 1
NB_PX = 2 ** (2 * N)

def load_images(path: str) -> np.ndarray:
    images = np.load(path)
    images = images / max(images.flatten()) * 255
    return images


def pixel_value_to_theta(pixel: float) -> float:
    return pixel / 255 * (np.pi / 2)


def theta_to_pixel_value(theta: float) -> int:
    return int(theta / (np.pi / 2) * 255)


def get_proba(counts: dict) -> dict:
    sums = sum(map(lambda x: x[1], counts.items()))
    return {key: value / sums for key, value in counts.items()}


def encode(image: np.ndarray) -> qiskit.QuantumCircuit:
    circuit = qiskit.QuantumCircuit(NB_QUBITS)

    # Get the theta values for each pixel
    image = image.flatten()
    thetas = [pixel_value_to_theta(pixel) for pixel in image]
    thetas += [0] * (NB_PX - NB_PX_IMG)

    # Apply Hadamard gates for all qubits except the last one
    for i in range(NB_QUBITS - 1):
        circuit.h(i)
    circuit.barrier()

    ry_qbits = list(range(NB_QUBITS))

    switches = [bin(0)[2:].zfill(NB_QUBITS)] + [
        bin(i ^ (i - 1))[2:].zfill(NB_QUBITS) for i in range(1, NB_PX)
    ]

    # Apply the rotation gates
    for i in range(NB_PX):
        theta = thetas[i]

        switch = switches[i]
        # Apply x gate to the i-th qubit if the i-th bit of the switch is 1
        for j in range(NB_QUBITS):
            if switch[j] == "1":
                circuit.x(j - 1)
        # TODO: Is this a 2-qubit gate?? -> If not we have to reformulate using 2-qubit gates only (RYGate + CNOT)
        # TODO: This method may be too slow: as such we have to compress the image by grouping pixels of the same intensity together
        c3ry = RYGate(2 * theta).control(NB_QUBITS - 1)
        circuit.append(c3ry, ry_qbits)

        circuit.barrier()

    circuit.measure_all()
    return circuit


def decode(counts: dict) -> np.ndarray:
    histogram = get_proba(counts)
    img = np.zeros(NB_PX)  # we have a square image

    for i in range(NB_PX):
        print(i)
        bin_str: str = np.binary_repr(i, width=NB_QUBITS - 1)
        print(bin_str)
        cos_str = "0" + bin_str[::-1]
        sin_str = "1" + bin_str[::-1]

        if cos_str in histogram:
            prob_cos = histogram[cos_str]
            theta = math.acos(np.clip(2**N * math.sqrt(prob_cos), 0, 1))
        else:
            prob_cos = 0

        # not needed?
        if sin_str in histogram:
            prob_sin = histogram[sin_str]
            theta = math.asin(np.clip(2**N * math.sqrt(prob_sin), 0, 1))
        else:
            prob_sin = 0

        img[i] = theta_to_pixel_value(theta)

    img = img[:NB_PX_IMG]
    return img.reshape(SIZE, SIZE)


def simulator(circuit: qiskit.QuantumCircuit) -> dict:
    # Simulate the circuit
    aer_sim = Aer.get_backend("aer_simulator")
    t_qc = transpile(circuit, aer_sim)
    qobj = assemble(t_qc, shots=16384)

    result = aer_sim.run(qobj).result()
    return result.get_counts(circuit)


def run_part1(image: np.ndarray) -> Union[qiskit.QuantumCircuit, np.ndarray]:
    circuit = encode(image)
    counts = simulator(circuit)
    img = decode(counts)
    return circuit, img

def count_gates(circuit: qiskit.QuantumCircuit) -> Dict[int, int]:
    """Returns the number of gate operations with each number of qubits."""
    return Counter([len(gate[1]) for gate in circuit.data])

def image_mse(image1,image2):
    # Using sklearns mean squared error:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    return mean_squared_error(image1, image2)

def grading(dataset):
    n=len(dataset)
    mse=0
    gatecount=0

    for data in dataset:
        circuit, image_re = run_part1(data['image'])
        # Count 2-qubit gates in circuit
        gatecount += count_gates(circuit)[2]
        
        # Calculate MSE
        mse += image_mse(data['image'],image_re)
        
    # Fidelity of reconstruction
    f = 1 - mse
    gatecount = gatecount / n

    # Score for Part 1
    return f * (0.999 ** gatecount)

if __name__ == "__main__":
    image = load_images("data/images.npy")[5]
    if image.max() != 0:
        image = image / image.max() * 255
    # plt.imshow(image, cmap='gray')
    # plt.show()

    # print(image)

    #image = np.array([0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 120])
    # image = np.array([128]*16)
    #image = image[:NB_PX]
    print(image)

    circuit = encode(image)

    # Simulate the circuit
    aer_sim = Aer.get_backend("aer_simulator")
    t_qc = transpile(circuit, aer_sim)
    qobj = assemble(t_qc, shots=16384)

    result = aer_sim.run(qobj).result()
    counts = result.get_counts(circuit)
    print(counts)
    print(len(counts))

    # Decode the histogram
    img = decode(get_proba(counts))
    print(img.flatten())
    plt.hist(img.flatten())
    plt.show()
