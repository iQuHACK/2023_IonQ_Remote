## FRQI image processing 28/01
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, Aer, transpile, assemble
from random import randint
from math import pi, atan, sqrt

n_scale = 3  # Use 2**n_scale colours
n_pixels = 3  # Square image with height = length = 2**n_pixels

test_image = [[randint(0, 2**n_scale - 1) / 2**n_scale * pi / 2
               for i in range(2**n_pixels)] for j in range(2**n_pixels)]
test_image[0][0] = 0
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

def encode(image, n_pix: int, n_shots=4096, draw=False):
    # Takes an image that is 2**n_pix by 2**n_pix pixels and encodes it
    # Pixel with intensity theta (theta <= pi/2) will be encoded as
    # cos theta |0yx> + sin theta |1yx>
    # Final state is measured n_shots times. Counts are returned as a dictionary.

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
            theta = image[-(i+1)][-(j+1)]

            # Apply X gates
            x_gate_location(qc_image, x_bin_coord, y_bin_coord, x_idx, y_idx)

            # Apply controlled rotation
            qc_image.mcry(2*theta, list(range(2*n_pixels)), 2*n_pixels)

            # Apply X gates
            x_gate_location(qc_image, x_bin_coord, y_bin_coord, x_idx, y_idx)

            qc_image.barrier()

    qc_image.measure_all()

    # Draw the circuit
    if draw:
        qc_image.draw()

    # Assemble and run circuit
    aer_sim = Aer.get_backend('aer_simulator')
    t_qc = transpile(qc_image, aer_sim)
    qobj = assemble(t_qc, shots=n_shots)
    result = aer_sim.run(qobj).result()
    counts = result.get_counts(qc_image)

    return counts


##
def decode(dic: dict, n_pix: int):
    # Decodes an image that has been encoded with encode
    # Image is 2**n_pix by 2**n_pix pixels

    # Array to reconstitute image
    decoded = [[0 for _ in range(2**n_pix)] for _ in range(2**n_pix)]

    for i in range(2**n_pix):  # loop over x coordinates
        for j in range(2**n_pix):  # loop over y coordinates
            x_str = bin(i)[2:].zfill(n_pix)
            y_str = bin(j)[2:].zfill(n_pix)
            key_0 = '0' + y_str + x_str
            key_1 = '1' + y_str + x_str

            # If no measurements of |0ji> / |1ji>, the corresponding pixel has full/minimum intensity
            # These cases are caught to avoid key errors
            if key_1 not in dic:
                decoded[j][i] = 0

            elif key_0 not in dic:
                decoded[j][i] = 1

            else:
                tan2_theta = dic[key_1] / dic[key_0]
                decoded[j][i] = atan(sqrt(tan2_theta))

    return np.array(decoded)


## Test random image
print(test_image)
print(decode(encode(test_image, n_pixels), n_pixels))

## Test image from file
images=np.load('data/images.npy')
test_image = images[0]*255
test_image = test_image[8:16, 8:16]
ans = decode(encode(test_image, 3), 3)
