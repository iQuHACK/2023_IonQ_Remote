## FRQI image processing 28/01

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, Aer, transpile, assemble
from random import randint
from math import pi

n_scale = 3  # Use 2**n_scale colours
n_pixels = 2  # Square image with height = length = 2**n_pixels

#test_image = [[randint(0, 2**n_scale - 1) / 2**n_scale * pi / 2 for i in range(n_pixels + 1)] for j in range(n_pixels + 1)]
test_image = [[0 for i in range(2**n_pixels)] for j in range(2**n_pixels)]

print(test_image)

##
def x_gate_location(q_circuit, x_coord: str, y_coord: str, x_idx, y_idx):
    # Given a set of coordinates, places X gates corresponding to those coordinates

    # Place X gates on x indicies
    for i in range(n_pixels):
        if x_coord[i] == '1':
            q_circuit.x(x_idx[n_pixels - i - 1])

    # Place X gates on y indicies
    for i in range(n_pixels):
        if y_coord[i] == '1':
            q_circuit.x(y_idx[n_pixels - i - 1])

def encode(image, n_pix: int):
    # Initialize the quantum circuit for the image
    # Pixel states
    x_idx = QuantumRegister(n_pix, 'x_idx')
    y_idx = QuantumRegister(n_pix, 'y_idx')

    # q state
    q = QuantumRegister(1, 'q')

    # create the quantum circuit for the image
    qc_image = QuantumCircuit(x_idx, y_idx, q)

    # Add Hadamard gates to the pixel positions
    for x in x_idx:
        qc_image.h(x)

    for y in y_idx:
        qc_image.h(y)

    # Apply controlled rotation gates
    for i in range(2**n_pixels):
        for j in range(2**n_pixels):
            # Obtain coordinates in binary, as well as intensity, registered as theta
            x_bin_coord = bin(j)[2:].zfill(n_pixels)
            y_bin_coord = bin(i)[2:].zfill(n_pixels)
            theta = image[i][j]

            # Apply X gates
            x_gate_location(qc_image, x_bin_coord, y_bin_coord, x_idx, y_idx)

            # Apply controlled rotation
            qc_image.mcry(2*theta, list(range(2*n_pixels)), 2*n_pixels)

            # Apply X gates
            x_gate_location(qc_image, x_bin_coord, y_bin_coord, x_idx, y_idx)

            qc_image.barrier()

    qc_image.measure_all()
    qc_image.draw()

    # Assemble and run circuit
    aer_sim = Aer.get_backend('aer_simulator')
    t_qc = transpile(qc_image, aer_sim)
    qobj = assemble(t_qc, shots=4096)
    result = aer_sim.run(qobj).result()
    counts = result.get_counts(qc_image)

    return counts

##
print(encode(test_image, n_pixels))

##

