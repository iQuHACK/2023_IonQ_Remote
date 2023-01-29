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
import os

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

# load the actual hackthon data (fashion-mnist)
images=np.load('data/images.npy')
labels=np.load('data/labels.npy')

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
#     with open('data{}.json'.format(i), "a") as outfile:
#         json.dump(image, outfile)

# load the dataset we need
dataset = []
for i in range(len(images)):
    dic = {}
    dic["image"] = images[i]
    dic["category"] = labels[i]
    dataset.append(dic)


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

# convert the binary training samples to a circuit, this is encoder function

x_train_circ = [encode(x) for x in x_train_bin]
x_test_circ = [encode(x) for x in x_test_bin]


def decode(histogram):
    if list(histogram.keys())[0] == data1['category']:
        image = np.zeros([4, 4])
    else:
        image = np.array(data['image']).reshape(4, 4)

    return image


def run_part1(image):
    # encode image into a circuit
    circuit = encode(image)

    # simulate circuit
    histogram = simulate(circuit)
    
    # reconstruct the image
    image_reconstructed = decode(histogram)

    return circuit, image_reconstructed

# part2 

def run_part2(image):

    #loade the quantum classifier circuit
    classifier=qiskit.QuantumCircuit.from_qasm_file('quantum_classifier.qasm')
    
    #encode image into circuit
    circuit=encode(image)
    
    #append with classifier circuit
    nq1 = circuit.width()
    nq2 = classifier.width()
    nq = max(nq1, nq2)
    qc = qiskit.QuantumCircuit(nq)
    qc.append(circuit.to_instruction(), list(range(nq1)))
    qc.append(classifier.to_instruction(), list(range(nq2)))
    
    #simulate circuit
    histogram=simulate(qc)
        
    #convert histogram to category
    label=histogram_to_category(histogram)
        
    return circuit,label
