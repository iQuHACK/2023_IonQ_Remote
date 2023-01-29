teamname = "Hilbert's Qerudites"
task = 'part 1'

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

#encode takes image as a 28x28 numpy array. 
def encode(image):
    q = qiskit.QuantumRegister(14)
    circuit = qiskit.QuantumCircuit(q)
    x=image[14:,:]
    x[x!=0.0]=1.0
    sum_x=list(np.sum(x,axis=1))
    sum_x=[x/28 for x in sum_x]
    for i in range(14):
        circuit.rx(sum_x[i]*np.pi,i)
    return circuit

def decode(histogram):
    if 1 in histogram.keys():
        image=np.full((28,28), 0)
    else:
        image=np.full((28,28), 1)
    return image

def run_part1(image):
    #encode image into a circuit
    circuit=encode(image)

    #simulate circuit
    histogram=simulate(circuit)

    #reconstruct the image
    image_re=decode(histogram)

    return circuit,image_re