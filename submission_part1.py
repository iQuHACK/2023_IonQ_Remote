# -*- coding: UTF-8 -*-

# Please see the iQuHACK.ipynb notebook for information and context!

import scipy.fftpack
import matplotlib.pyplot as plt
import qiskit
import numpy as np


team_name = 'HaQ'
task = 'part 1'


# Provided by IonQ

def simulate(circuit: qiskit.QuantumCircuit) -> dict:
    """Simulate the circuit, give the state vector as the result."""
    backend = qiskit.BasicAer.get_backend('statevector_simulator')
    job = qiskit.execute_function.execute(circuit, backend)
    result = job.result()
    state_vector = result.get_statevector()

    histogram = dict()
    for i in range(len(state_vector)):
        population = abs(state_vector[i]) ** 2
        if population > 1e-9:
            histogram[i] = population

    return histogram

# Helper function


def bin_array(num, m):
    """Convert a positive integer num into an m-bit bit vector"""
    return np.array(list(np.binary_repr(num).zfill(m))).astype(np.int8)


def compress(image, size, eps=0):
    """Converts an image into its 2D-DCT, with some normalization."""
    a = scipy.fftpack.dctn(image)
    comp_freqs = a[:size, :size]
    max_freq = np.max(np.abs(comp_freqs))
    return comp_freqs/max_freq, max_freq


def decompress(freqs, scale, size):
    """Converts an image in 2D-DCT space back to real space."""
    new_freqs = np.zeros((size, size))
    s = freqs.shape[0]
    new_freqs[:s, :s] = freqs/scale
    reconstructed_image = scipy.fftpack.idctn(new_freqs)
    return reconstructed_image


def encode(image, eps=1e-6):
    if image.shape[0] != image.shape[1]:
        raise ValueError('Image must be square.')
    if np.min(image) < 0:
        raise ValueError('Image cannot have negative entries.')

    image_side = image.shape[0]
    n = image_side.bit_length() - 1

    q = qiskit.QuantumRegister(2*n+1)
    ct = qiskit.QuantumCircuit(q)

    # The DCT maps onto [-1, 1]; since we'll be estimating probabilities,
    # we need the domain to be [0, 1].
    image /= np.max(np.abs(image))
    image += 1
    image /= 2

    ct.h([qi for qi in q][:-1])

    for i in range(len(image)):
        for j in range(len(image[0])):

            aux_i = bin(i)[2:].zfill(n)
            aux_j = bin(j)[2:].zfill(n)

            theta = (image[i, j] - 0.5)*np.pi/2

            if abs(theta) > eps:
                rotation = qiskit.circuit.library.RYGate(2.*theta)
                rotation = rotation.control(num_ctrl_qubits=2*n,
                                            ctrl_state=(aux_i + aux_j))
                ct.append(rotation, q)
    ct.ry(2 * np.pi/4, q[-1])

    return ct


def decode(histogram, image_side):
    # This function considers that the histogram is actually the probabilities
    # computed via the wavefunction.
    index_reg_qubits = int(np.ceil(np.log2(image_side)))
    total_qubits = 2*index_reg_qubits + 1
    n_pixels = image_side**2

    # The matix to which we'll place our reconstructed image...
    data = np.zeros((image_side, image_side))

    for key in range(2**total_qubits):
        arr = bin_array(key, total_qubits)
        if arr[0] == 0:
            arr_1, arr_2 = np.split(arr[1:], 2)
            c_1 = arr_1[::-1].dot(2**np.arange(arr_1.size)[::-1])
            c_2 = arr_2[::-1].dot(2**np.arange(arr_2.size)[::-1])
            data[c_2, c_1] = (
                4/np.pi) * np.arccos(np.sqrt(histogram.get(key, 0.) * n_pixels)) - 1
            #print(key, arr_1, arr_2, c_1, c_2)

    return data


def encoder(image):
    circuit = encode(image, eps=5e-2)
    return circuit


def decoder(histogram, freq_shape, scale, data_shape):
    freqs_re = decode(histogram, freq_shape)
    image_re = decompress(freqs_re, scale, data_shape)
    return image_re


def run_part1(image):
    freqs, scale = compress(image, 2)
    circuit = encode(freqs)
    histogram = simulate(circuit)
    return decoder(histogram, freqs.shape[0], scale, image.shape[0])
