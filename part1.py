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
NB = 2**(N*2)

def load_images(path: str) -> np.ndarray:
    images = np.load(path)
    images = images / max(images.flatten()) * 255
    return images

def pixel_value_to_theta(pixel: float) -> float:
    return pixel / 255 * (np.pi/2)

def theta_to_pixel_value(theta: float) -> int:
    return int(theta / (np.pi/2) * 255)

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

    switches = [bin(0)[2:].zfill(NB_QUBITS)] + [bin(i ^ (i-1))[2:].zfill(NB_QUBITS) for i in range(1, NB)]

    # Apply the rotation gates
    for i in range(NB):
        theta = thetas[i]

        switch = switches[i]
        # Apply x gate to the i-th qubit if the i-th bit of the switch is 1
        for j in range(NB_QUBITS):
            if switch[j] == '1':
                circuit.x(j-1)
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
    img = np.zeros(NB)
    print(histogram)

    for i in range(NB):
        print(i)
        bin_str = np.binary_repr(i, width=NB_QUBITS - 1)
        print(bin_str)
        cos_str = "0" + bin_str[::-1]
        sin_str = "1" + bin_str[::-1]

        n0 = 1

        if cos_str in histogram:
            prob_cos = histogram[cos_str]**2
        else:
            prob_cos = 0

        # not needed?
        if sin_str in histogram:
            prob_sin = histogram[sin_str]**2
        else:
            prob_sin = 0

        print(n0, cos_str, sin_str)
        print(prob_cos, prob_sin)
        theta = math.acos(prob_cos)
        theta = np.pi/2*prob_sin/(prob_cos + prob_sin)
        print(theta)

        img[i] = theta_to_pixel_value(theta)

    return img#.reshape(SIZE, SIZE)

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
    #image = np.array([0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 255, 0, 120])
    image = np.array([128]*16)
    image = image[:NB]
    print(2**SIZE-1)
    circuit = frqi_encode(image)
    # Simulate the circuit
    aer_sim = Aer.get_backend('aer_simulator')
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