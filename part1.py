import qiskit
import numpy as np
import math

SIZE = 28
N = math.ceil(math.log2(SIZE))
NB_QUBITS = 2*N + 1

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
    
    # Apply the rotation gates wrt theta...
    
    
    # Print the circuit
    print(circuit)
    return circuit

if __name__ == "__main__":
    image = load_images('data/images.npy')[0]
    circuit = frqi_encode(image)