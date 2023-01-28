import qiskit
from qiskit.execute_function import execute
from qiskit import BasicAer
from qiskit import transpile, assemble
import numpy as np
import pickle
import json
from collections import Counter
from sklearn.metrics import mean_squared_error
from typing import Dict, List, Tuple
import os
import cv2

############################
### ENCODE AND DECODE DATA ###
############################


def encode_qiskit(image):
    q = qiskit.QuantumRegister(3)
    circuit = qiskit.QuantumCircuit(q)
    if image[0][0] == 0:
        circuit.rx(np.pi, 0)
    return circuit


def decode(histogram):
    if 1 in histogram.keys():
        image = [[0, 0], [0, 0]]
    else:
        image = [[1, 1], [1, 1]]
    return image

################################################
############## HELPERS #########################
################################################


def image_mse(img1, img2):
    return mean_squared_error(img1, img2)


def count_gates(circuit: qiskit.QuantumCircuit) -> Dict[int, int]:
    """ finds num of gate operations with each num of qubits """
    return Counter([len(gate[1]) for gate in circuit.data])

############################
### SIMULATION AND HISTOGRAM ###
############################


def simulate(circuit: qiskit.QuantumCircuit) -> Dict:
    """Simulate circuit given state vector"""
    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(circuit, backend)
    result = job.result()
    state_vector = result.get_statevector(circuit)

    histogram = dict()
    for i in range(len(state_vector)):
        population = abs(state_vector[i]) ** 2
        if population > 1e-9:
            histogram[i] = population

    return histogram


def histogram_to_cat(histogram):
    assert abs(sum(histogram.values()) - 1) < 1e-8
    positive = 0
    for key in histogram.keys():
        digits = bin(int(key))[2:].zfill(20)
        if digits[-1] == '0':
            positive += histogram[key]

    return positive


################################################
############### MAIN ###########################
################################################


# load data
files = os.listdir("mock_data")
dataset = []
dataset_labels = []
dataset = np.load('data/images.npy')
dataset_labels = np.load('data/labels.npy')
# for file in files:
#     with open('mock_data/'+file, "r") as infile:
#         loaded = json.load(infile)
#         dataset.append(loaded)


n = len(dataset)

# for data in dataset:
#     circuit = encode_qiskit(data['image'])
#     histogram = simulate(circuit)
#     gate_counts = count_gates(circuit)[2]
#     image_re = decode(histogram)

# https://qiskit.org/textbook/ch-applications/image-processing-frqi-neqr.html
scaled_data = 255*255*dataset[0]
# convert to binary

# one image
bin_data = [format(int(scaled_data[i][j]), '08b')
            for i in range(28) for j in range(28)]
print(bin_data)


# douwnsampling
# https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
image = cv2.imread('data/images.npy')
image = cv2.cvtColor(numpy.asarray(img), cv2.COLOR_BGR2GRAY)
cv2.imshow("OpenCV image", image)

# 1. intialize qubits
theta = 0  # all pixels are black
idx = qiskit.QuantumRegister(3, 'idx')
intensity = qiskit.QuantumRegister(8, 'intensity')
cr = qiskit.ClassicalRegister(10, 'cr')

qc_image = qiskit.QuantumCircuit(intensity, idx, cr)
num_qubits = qc_image.num_qubits

# add Hadamard gates
qc_image.h(8)
qc_image.h(9)
qc_image.barrier()

# 2. represent grayscaled image

# encode first pixel
for idx in range(num_qubits):
    qc_image.i(idx)

qc_image.barrier()

# 3. encode image
