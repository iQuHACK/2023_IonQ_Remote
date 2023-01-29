# Team Name: quantum potato
# Team Members: Max C., Shivane D., Adelina C.
# Task: Part 1 and Part 2

import tensorflow as tf
import sympy
import cirq
import numpy as np
import pickle
import json
import os
import sys
from collections import Counter
from sklearn.metrics import mean_squared_error
import collections
import seaborn as sns

if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    data_path = 'data'

#define utility functions

def simulate(circuit: cirq.Circuit) -> dict:
    """This function simulates a Cirq circuit (without measurement) and outputs results in the format of histogram.
    """
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    
    state_vector=result.final_state_vector
    
    histogram = dict()
    for i in range(len(state_vector)):
        population = abs(state_vector[i]) ** 2
        if population > 1e-9:
            histogram[i] = population
    
    return histogram


def histogram_to_category(histogram):
    """This function takes a histogram representation of circuit execution results, and processes into labels as described in
    the problem description."""
    #assert abs(sum(histogram.values())-1)<1e-8
    positive=0
    for key in histogram.keys():
        digits = bin(int(key))[2:].zfill(20)
        if digits[-1]=='0':
            positive+=histogram[key]
        
    return positive

def count_gates(circuit: cirq.Circuit):
    """Returns the number of 1-qubit gates, number of 2-qubit gates, number of 3-qubit gates...."""
    counter=Counter([len(op.qubits) for op in circuit.all_operations()])
    
    #feel free to comment out the following two lines. But make sure you don't have k-qubit gates in your circuit
    #for k>2
    #for i in range(2,20):
    #    assert counter[i]==0
        
    return counter
EEEE = tf.keras.losses.MeanSquaredError()

def image_mse(image1, image2):
    # Using sklearns mean squared error:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    return EEEE(image1, image2).numpy()
#    return mean_squared_error(image1, image2)

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
def encode(image):
    THRESHOLD = 0.0
    image = image.reshape(*(28, 28, 1))
    image = tf.image.resize(image, (4,4)).numpy()
    for i in range(len(image)):
        for j in range(len(image[i])):
            image[i][j] = [1.0 if image[i][j] > THRESHOLD else 0.0]
    values = np.ndarray.flatten(image)
    qubits = cirq.GridQubit.rect(4,4)
    circuit = cirq.Circuit()
    for i, val in enumerate(values):
        if val: circuit.append(cirq.X(qubits[i]))
    return circuit

def decode(hist):
    trash = []
    for pixel in range(16):
        bit_str = f"{pixel:02b}"
        n_ones = hist.get(int('1'+bit_str), 0.0)
        n_zeros = hist.get(int('0'+bit_str), 0.0)
        if n_ones == 0 and n_zeros == 0:
            pixel_value = 0
        else:
            pixel_value = n_ones / (n_ones + n_zeros)
        trash.append(pixel_value)
    trash = np.array(trash)
    trash = trash.reshape(*(4, 4, 1))
    trash = tf.image.resize(trash, (28,28)).numpy()
    return trash

def run_part1(image):
    #encode image into a circuit
    circuit=encode(image)

    #simulate circuit
    histogram=simulate(circuit)

    #reconstruct the image
    image_re=decode(histogram)

    return circuit, image_re

class CircuitLayerBuilder():
    def __init__(self, data_qubits, readout):
        self.data_qubits = data_qubits
        self.readout = readout

    def add_layer(self, circuit, gate, prefix):
        for i, qubit in enumerate(self.data_qubits):
            #symbol = sympy.Symbol(prefix + '-' + str(i))
            circuit.append(gate(qubit, self.readout) ** (np.random.random() * np.pi))

def create_quantum_model():
    """Create a QNN model circuit and readout operation to go along with it."""
    data_qubits = cirq.GridQubit.rect(4, 4)  # a 4x4 grid.
    readout = cirq.GridQubit(-1, -1)         # a single qubit at [-1,-1]
    circuit = cirq.Circuit()

    # Prepare the readout qubit.
    circuit.append(cirq.X(readout))
    circuit.append(cirq.H(readout))

    builder = CircuitLayerBuilder(
        data_qubits = data_qubits,
        readout=readout)

    # Then add layers (experiment by adding more).
    builder.add_layer(circuit, cirq.XX, "xx1")
    #builder.add_layer(circuit, cirq.ZZ, "zz1")

    # Finally, prepare the readout qubit.
    circuit.append(cirq.H(readout))

    return circuit, cirq.Z(readout)

def hinge_accuracy(y_true, y_pred):
    y_true = tf.squeeze(y_true) > 0.0
    y_pred = tf.squeeze(y_pred) > 0.0
    result = tf.cast(y_true == y_pred, tf.float32)

    return tf.reduce_mean(result)

def run_part2(image):
    model_circuit, model_readout = create_quantum_model()
    circuit = encode(image)
    
    #append with classifier circuit
    circuit.append(model_circuit)
    
    #simulate circuit
    histogram=simulate(circuit)
        
    #convert histogram to category
    label=histogram_to_category(histogram)
    
    #thresholding the label, any way you want
    if label>0.5: label=1
    else: label=0
        
    return circuit,label

############################
#      END YOUR CODE       #
############################

test()
