teamname = 'Heisen_bugs'
task = 'part 1'

import os
import qiskit
from qiskit import quantum_info
from qiskit.execute_function import execute
from qiskit import BasicAer
import numpy as np
import pickle
import json
import os
from collections import Counter
from sklearn.metrics import mean_squared_error
from typing import Dict, List
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_quantum as tfq

import cirq
import sympy
import numpy as np
import seaborn as sns
import collections

# visualization tools
%matplotlib inline
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit


# load and split the dataset

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))

# load the dataset from slices onto a varaible called dataset
train, test = tf.keras.datasets.fashion_mnist.load_data()
images, labels = train
images = images/255

# replace 4,4 with 8,8 or higher for better score
x_train_small = tf.image.resize(x_train, (4, 4)).numpy()
x_test_small = tf.image.resize(x_test, (4, 4)).numpy()

# check if data passes threshold or not
THRESHOLD = 0.5

x_train_bin = np.array(x_train_small > THRESHOLD, dtype=np.float32)
x_test_bin = np.array(x_test_small > THRESHOLD, dtype=np.float32)

# create dataset we need
# for i in range(len(x_test_bin)):
#     x_test_bin[i].reshape(4, 4)
#     value = x_test_bin[i].tolist()

#     image = {"image": value, "category": int(y_test[i])}
#     with open('hack2/data{}.json'.format(i), "a") as outfile:
#         json.dump(image, outfile)

# load the dataset we need
dataset = []
for i in range(len(images)):
    dic = {}
    dic["image"] = images[i]
    dic["category"] = labels[i]
    dataset.append(dic)

def count_gates(circuit: cirq.Circuit):
    """Returns the number of 1-qubit gates, number of 2-qubit gates, number of 3-qubit gates...."""
    counter = Counter([len(op.qubits) for op in circuit.all_operations()])

    # feel free to comment out the following two lines. But make sure you don't have k-qubit gates in your circuit
    # for k>2
    for i in range(2, 20):
        assert counter[i] == 0

    return counter


def image_mse(image1, image2):
    return mean_squared_error(image1, image2)


def encode(image):
    """Encode truncated classical image into quantum datapoint."""
    image = np.array(image)
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4, 4)
    circuit = cirq.Circuit()
    for i, value in enumerate(values):
        if value:
            circuit.append(cirq.X(qubits[i]))
    return circuit


def circuit_to_histogram(circuit):
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)

    state_vector = result.final_state_vector

    histogram = dict()
    for i in range(len(state_vector)):
        population = abs(state_vector[i]) ** 2
        if population > 1e-9:
            histogram[i] = population

    return histogram

# convert the binary training samples to a circuit, this is encoder function

x_train_circ = [encode(x) for x in x_train_bin]
x_test_circ = [encode(x) for x in x_test_bin]


def decode(histogram):
    if list(histogram.keys())[0] == data1['category']:
        image = np.zeros([4, 4])
    else:
        image = np.array(data1['image']).reshape(4, 4)

    return image


def run_part1(image):
    # encode image into a circuit
    circuit = encode(image)

    # simulate circuit
    histogram = circuit_to_histogram(circuit)
    
    # reconstruct the image
    image_reconstructed = decode(histogram)

    return circuit, image_reconstructed

# x_train_circ = [ encode(x) for x in x_train_bin]
# x_test_circ = [ encode(x) for x in x_test_bin]

# OPTIONAL : GRADING CODE how we grade your submission

# n = len(dataset)
# mse = 0
# gatecount = 0

# for data in dataset:
#     # encode image into circuit
#     circuit, image_re = run_part1(data1['image'])

#     # count the number of 2qubit gates used
#     gatecount += count_gates(circuit)[2]

#     #data1['image'] = np.array(data1['image']).reshape(4, 4)
#     #image_re.reshape(4, 4)

#     # calculate mse
#     mse += image_mse(data['image'], image_re)

# # fidelity of reconstruction
# f = 1 - mse
# gatecount = gatecount / n

# # score for part1
# print(f * (0.999 ** gatecount))
