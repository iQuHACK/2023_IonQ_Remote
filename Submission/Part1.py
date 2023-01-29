"""

Part one submission.

Contains the simulate(circuit: qiskit.QuantumCircuit), encode(image), decode(histogram) and run_part1(image) functions.

""" 
import qiskit 
from qiskit import QuantumCircuit, QuantumRegister
from qiskit import transpile, assemble
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
from math import ceil

#Simulate
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



#encode
def encode(image):
    epsilon = 0.1/255 #background is sometimes not exactly 0
    
    if len(image) < 28:
        rows_to_check = [0, 1]
    else:
        rows_to_check = [1, 7, 14, 21, 26] #rows we use to get info
    q = 2*len(rows_to_check)
          
    qubits = QuantumRegister(q,'qubits')
    qc = QuantumCircuit(qubits)
    for row in rows_to_check:
        for i in range(len(image[0])):
            if image[row][i] >= epsilon:
                qc.rx(np.pi/len(image[0]),qubits[rows_to_check.index(row)]) #should represent % non-background values for a row in rows_to_check (first |rows_to_check| qubits)
                #check amount of "segments" (between 0 and 3) in each row in rows_to_check
                if i == 0:
                    qc.rx(np.pi/3,qubits[rows_to_check.index(row)+len(rows_to_check)])#starts with non-background e.g. [[1,1],[1,1]]
                else:
                    if image[row][i-1] < epsilon:
                        qc.rx(np.pi/3,rows_to_check.index(row)+len(rows_to_check))
    qc=transpile(qc) 
    return qc



#decode
def decode(histogram):
    #need to know size of picture 
    if len(histogram) <= 4: #mock data
        if 1 in histogram.keys():
            return [[0,0],[0,0]]
        else:
            return [[1,1],[1,1]]
    else:
        rows_to_check = 5
        decoded_rows = []
        decoded_qubit_values = [0]*2*rows_to_check
        for key in histogram.keys():
            #find qubit that corresponds to your row to find
            binary_key = str(bin(key)[2:].zfill(len(decoded_qubit_values)))
            for i in range(len(binary_key)):
                if binary_key[i] == '1':
                    decoded_qubit_values[i] += histogram[key]
        for i in range(len(decoded_qubit_values)//2):
            #non-background pixels in row
            r = ceil(decoded_qubit_values[i*2]*28)
            #segments
            s = int(decoded_qubit_values[i*2+1]*3+0.1)
            #construct row without info on intensity
            if s == 0:
                l = 0 
            else:
                l = r//s
            row = []
            for i in range(s):
                row += [0.5/255]*l + [0]
            row = row[:-1]
            decoded_row = row + [0]*int((28-r-max(0,s-1))/2)
            decoded_row = [0]*(28-len(decoded_row))+decoded_row
            decoded_rows.append(decoded_row)
        decoded_image = []
        for i in range(28):
            if i <4:
                index = 0
            elif i < 11:
                index = 1
            elif i < 17:
                index = 2
            elif i < 25:
                index = 3
            else:
                index = 4
            decoded_image.append(decoded_rows[index])
        return np.array(decoded_image)
            
            

#run_part1
def run_part1(image):
    #encode image into a circuit
    circuit=encode(data['image'])

    #simulate circuit
    histogram=simulate(circuit)

    #reconstruct the image
    image_re=decode(histogram)

    return circuit, image_re