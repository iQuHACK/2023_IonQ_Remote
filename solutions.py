import qiskit
import numpy as np
from math import acos, floor, sqrt
from qiskit import quantum_info
from qiskit.execute_function import execute
from qiskit import BasicAer
import numpy as np
import pickle
import json
import os
from collections import Counter
from typing import Dict, List
import matplotlib.pyplot as plt


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
    """This function take a histogram representations of circuit execution results, and process into labels as described in 
    the problem description."""
    assert abs(sum(histogram.values())-1)<1e-8
    positive=0
    for key in histogram.keys():
        digits = bin(int(key))[2:].zfill(20)
        if digits[-1]=='0':
            positive+=histogram[key]
        
    return positive


def encode(image):
    aim = np.zeros((7,7))
    for x in range(7):
        for y in range(7):
            avg_val = 0
            counter = 0
            for l in range(4):
                for k in range(4):
                    avg_val += image[4*x+l][4*y+k]
                    counter += 1
            aim[x][y] = avg_val / counter

    num_qubits = 16
    q = qiskit.QuantumRegister(num_qubits)
    circuit = qiskit.QuantumCircuit(q)
    
    im = aim.flatten()[:-1]
    
    should_continue = True
    
    for i in range(num_qubits):
        pixel, im = im[0], im[1:]
        circuit.rx(acos(pixel)*2, i)
        pixel, im = im[0], im[1:]
        circuit.rz(acos(pixel)*2, i)
        pixel, im = im[0], im[1:]
        circuit.rx(acos(pixel)*2, i)
    return circuit

def decode(histogram):
    return np.zeros((28,28))

def run_part1(image):
    #encode image into a circuit
    circuit=encode(image)

    #simulate circuit
    histogram=simulate(circuit)

    #reconstruct the image
    image_re=decode(histogram)

    return circuit,image_re

def run_part2(image):

    #loade the quantum classifier circuit
    with open('quantum_classifier.pickle', 'rb') as f:
        classifier=pickle.load(f)
    
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
