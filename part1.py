import qiskit
from qiskit import quantum_info
from qiskit.execute_function import execute
from qiskit import BasicAer
import numpy as np
import pickle
import json
import os
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from typing import Dict, List
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister


#this simulate a circuit (without measurement) and output results in the format of histogram.
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


def count_gates(circuit: qiskit.QuantumCircuit) -> Dict[int, int]:
    """Returns the number of gate operations with each number of qubits."""
    counter = Counter([len(gate[1]) for gate in circuit.data])
    #feel free to comment out the following two lines. But make sure you don't have k-qubit gates in your circuit
    #for k>2
    # for i in range(2,20):
    #     assert counter[i]==0
        
    return counter
def image_mse(image1,image2):
    # Using sklearns mean squared error:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    return mean_squared_error(image1, image2)


def binary(decimal):
    return bin(decimal)[2:]

def roundint(number):
    return int(np.rint(number))

def unnest_matrix(matrix):
    return [col for row in matrix for col in row]

def norm(matrix):
    return sum([np.abs(item)**2 for item in unnest_matrix(matrix)])



##################################
def encode(image):
    
    Norm = norm(image)
    unnested_image = unnest_matrix(image)
    
    l = len(image)
    L = l**2
    
    n=int(np.ceil(np.log2(L)))
    N=2**n
    
    initial_state = [0]*N
    
    for index in range(N-L,N):
        initial_state[index] = unnested_image[index-(N-L)]/np.sqrt(Norm)
    
    qr = qiskit.QuantumRegister(n)
    qc = qiskit.QuantumCircuit(qr)

    qc.initialize(initial_state,qr)
    return qc


def decode(hist):
    
    l=28
    L=l**2
    
    n=10
    N=2**n
    
    image = [0]*L
    for key,value in H.items():
        image[key-(N-L)] = value
    
    matrix = [image[i:i+l] for i in range(0, L, l)]
    return matrix    



def run_part1(image):
    #encode image into a circuit
    circuit=encode(image)

    #simulate circuit
    histogram=simulate(circuit)

    #reconstruct the image
    image_re=decode(histogram)

    return circuit,image_re