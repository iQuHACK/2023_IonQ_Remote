import qiskit
from qiskit import quantum_info
from qiskit.execute_function import execute
from qiskit import BasicAer
import numpy as np
from collections import Counter
from sklearn.metrics import mean_squared_error
from typing import Dict, List
import matplotlib.pyplot as plt
import os

import cirq
import sympy
import numpy as np

import tensorflow as tf

import matplotlib.pyplot as plt

import cirq
import cv2
import math
from qiskit import QuantumCircuit, Aer
from qiskit import transpile, assemble
from qiskit.visualization import plot_histogram
from math import pi


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range.
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))

# load the dataset from slices onto a varaible called dataset
train, test = tf.keras.datasets.fashion_mnist.load_data()
images, labels = train
images = images/255

# check if data passes threshold or not
# THRESHOLD = 0.5

# x_train_bin = np.array(x_train_small > THRESHOLD, dtype=np.float32)
# x_test_bin = np.array(x_test_small > THRESHOLD, dtype=np.float32)


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
    image /= 255
    image = cv2.resize(image,(2,2))
    image = list(image.reshape(4))
    qc = QuantumCircuit(3)
    qc.h(0)
    qc.h(1)
    x_dict ={
        0: [],
        1: [1],
        2: [1,0],
        3: [1]
        
    }

    for i in range(len(image)):
        theta = math.pi*image[i]/2
        for j in x_dict[i]:
            qc.x(j)
        qc.cry(theta,0,2)
        qc.cx(0,1)
        qc.cry(-theta,1,2)
        qc.cx(0,1)
        qc.cry(theta,1,2)
    return qc

def simulate(circuit:QuantumCircuit):
    circuit.measure_all()
    aer_sim = Aer.get_backend('aer_simulator')
    t_qc = transpile(circuit, aer_sim)
    qobj = assemble(t_qc, shots=4096)
    result = aer_sim.run(qobj).result()
    counts = result.get_counts(circuit)
    for item in counts:
        counts[item] /= 4096
    return counts

# convert the binary training samples to a circuit, this is encoder function


def decode(histogram):
    
    order = ["00","01","10","11"]
    values_dict = {}
    for item in histogram:
        
        if item[0] == "1":
            
            values_dict[f"{item[1:]}"] = math.asin(min(math.sqrt(histogram[item]*4),1))*(2/math.pi)
        if item[0] == "0":
            
            values_dict[f"{item[1:]}"] = math.acos(min(math.sqrt(histogram[item]*4),1))*(2/math.pi)
        
    
    answer = []
    for item in order:
        answer.append(values_dict[item])
        
    image = (np.array(answer)).reshape(2,2)
    image = cv2.resize(image,(28,28))

    return image
# convert the binary training samples to a circuit, this is encoder function

# x_train_circ = [encode(x) for x in x_train]
# x_test_circ = [encode(x) for x in x_test]




def run_part1(image):
    # encode image into a circuit
    circuit = encode(image)

    # simulate circuit
    histogram = simulate(circuit)
    
    # reconstruct the image
    image_reconstructed = decode(histogram)

    return circuit, image_reconstructed

def count_gates(circuit: qiskit.QuantumCircuit) -> Dict[int, int]:
    """Returns the number of gate operations with each number of qubits."""
    counter = Counter([len(gate[1]) for gate in circuit.data])
    print("-------------------")
    for gate in circuit.data:
        
        if len(gate[1])>2:
            print(gate)
    print("------------")
    print(counter)
        
    return counter

def image_mse(image1, image2):
    return mean_squared_error(255*image1, 255*image2)

n = len(dataset)
mse = 0
gatecount = 0

i = 0
for data in dataset:
    i+=1
    # encode image into circuit
    circuit, image_re = run_part1(data['image'])

    # count the number of 2qubit gates used
    gatecount += count_gates(circuit)[2]

    

    # calculate mse
    mse += image_mse(data["image"], image_re)

# fidelity of reconstruction
f = 1 - mse
gatecount = gatecount / n

# score for part1
print(f * (0.999 ** gatecount))
