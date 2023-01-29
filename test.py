import qiskit
from qiskit import quantum_info
from qiskit.execute_function import execute
from qiskit import BasicAer
import numpy as np
import pickle
import json
import os
import sys
from collections import Counter
from sklearn.metrics import mean_squared_error
from typing import Dict, List
import matplotlib.pyplot as plt

if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    data_path = '.'

#define utility functions

def simulate(circuit: qiskit.QuantumCircuit) -> dict:
    """Simulate the circuit, give the state vector as the result."""
    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(circuit, backend)
    result = job.result()
    state_vector = result.get_statevector()
    
    histogram = dict()
    for i in range(len(state_vector)):
        population = abs(state_vector[i]) ** 2
        if population > 1e-9:
            histogram[i] = population
    
    return histogram


def histogram_to_category(histogram):
    """This function takes a histogram representation of circuit execution results, and processes into labels as described in
    the problem description."""
    assert abs(sum(histogram.values())-1)<1e-8
    positive=0
    for key in histogram.keys():
        digits = bin(int(key))[2:].zfill(20)
        if digits[-1]=='0':
            positive+=histogram[key]
        
    return positive

def count_gates(circuit: qiskit.QuantumCircuit) -> Dict[int, int]:
    """Returns the number of gate operations with each number of qubits."""
    counter = Counter([len(gate[1]) for gate in circuit.data])
    #feel free to comment out the following two lines. But make sure you don't have k-qubit gates in your circuit
    #for k>2
    for i in range(3,20):
        assert counter[i]==0
        
    return counter


def image_mse(image1,image2):
    # Using sklearns mean squared error:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    return mean_squared_error(image1,image2)

def test():
    #load the actual hackthon data (fashion-mnist)
    images=np.load(data_path+'/images.npy')
    labels=np.load(data_path+'/labels.npy')
    
    #test part 1

    n=len(images)
    mse=0
    gatecount=0

    for image in images:
        #encode image into circuit
        circuit,image_re=run_part1(image)
        image_re = np.asarray(image_re)

        #count the number of 2qubit gates used
        gatecount+=count_gates(circuit)[2]

        #calculate mse
        mse+=image_mse(image,image_re)

    #fidelity of reconstruction
    f=1-mse/n
    gatecount=gatecount/n

    #score for part1
    score_part1=f*(0.999**gatecount)
    
    #test part 2
    
    score=0
    gatecount=0
    n=len(images)

    for i in range(n):
        #run part 2
        circuit,label=run_part2(images[i])

        #count the gate used in the circuit for score calculation
        gatecount+=count_gates(circuit)[2]

        #check label
        if label==labels[i]:
            score+=1
    #score
    score=score/n
    gatecount=gatecount/n

    score_part2=score*(0.999**gatecount)
    
    print(score_part1, ",", score_part2, ",", data_path, sep="")


############################
#      YOUR CODE HERE      #
############################
import scipy.fftpack
import matplotlib.pyplot as plt
import qiskit
import numpy as np


team_name = 'HaQ'
task = 'part 1'


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


def encoder(image, eps=1e-6):
    if image.shape[0] != image.shape[1]:
        raise ValueError('Image must be square.')

    image_side = image.shape[0]
    n = int(np.ceil(np.log2(image_side)))

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
                key = (aux_i + aux_j)[::-1]
                rotation = rotation.control(num_ctrl_qubits=2*n,
                                            ctrl_state=key)
                ct.append(rotation, q)
    ct.ry(2 * np.pi/4, q[-1])

    return ct


def decoder(histogram, image_side):
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

    return data


def encode(image, return_scale=False):
    freqs, scale = compress(image, 4)
    circuit = encoder(freqs, eps=5e-2)
    circuit = qiskit.transpile(circuit, basis_gates=[
                               'u', 'cx'], optimization_level=3)
    if return_scale:
        return scale, circuit
    else:
        return circuit


def decode(histogram, scale, data_shape):
    freqs_re = decoder(histogram, 4)
    image_re = decompress(freqs_re, scale, data_shape)
    return image_re


def run_part1(image):
    scale, circuit = encode(image, return_scale=True)
    histogram = simulate(circuit)
    return circuit, decode(histogram, scale, image.shape[0])


def run_part2(image):
    # load the quantum classifier circuit
    classifier = qiskit.QuantumCircuit.from_qasm_file(
        'submission_classifier.qasm')

    # encode image into circuit
    circuit = encode(image)

    # append with classifier circuit
    nq1 = circuit.width()
    nq2 = classifier.width()
    nq = max(nq1, nq2)
    qc = qiskit.QuantumCircuit(nq)
    qc.append(circuit.to_instruction(), list(range(nq1)))
    qc.append(classifier.to_instruction(), list(range(nq2)))

    # simulate circuit
    histogram = simulate(qc)

    # convert histogram to category
    label = histogram_to_category(histogram)

    if label > 0.5:
        label = False
    else:
        label = True

    return circuit, label

############################
#      END YOUR CODE       #
############################

test()
