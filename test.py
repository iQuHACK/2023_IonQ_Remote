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
from qiskit import QuantumCircuit, assemble, Aer
from qiskit.visualization import plot_bloch_multivector, plot_histogram, plot_bloch_vector, array_to_latex, plot_state_qsphere
from math import pi, sqrt
import pickle

sim = Aer.get_backend('aer_simulator') 

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
    #for i in range(2,20):
    #    assert counter[i]==0
        
    return counter


def image_mse(image1,image2):
    #Using sklearns mean squared error:https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    return mean_squared_error(255*image1,255*image2)

def test():
    #load the actual hackthon data (fashion-mnist)
    images=np.load('data/images.npy')
    labels=np.load('data/labels.npy')
    
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
def normalize_image(image):
    sum_sq = 0
    for row in image:
        for pixel in row:
            sum_sq += pixel ** 2
            
    return image / np.sqrt(sum_sq)
    
def encoder(image):
    image = normalize_image(image)
    n = 10
    qc = QuantumCircuit(n)
    
    for i in range(n):
        qc.h(i)
    
    # 10 qubits, cada uno tiene 2 estados == 1024 estados a disposicion
    size = len(image)
    initial_state = np.zeros(2**n)
    counter = 0
    
    for i in range(size):
        for j in range(size):
            initial_state[counter] = image[i][j]
            counter += 1
            
    qc.initialize(initial_state)
    qc.save_statevector()
       
    return qc

def decoder(counts):
    size = 28
    reconstruction = np.zeros([size, size])

    binaries = list(counts.keys())
    decimals = list(map(lambda b: int(b, 2), binaries))

    for b in list(counts.keys()):
        d = int(b, 2)
        i = int(np.floor(d / 28))
        j = int(d % 28)
        reconstruction[i][j] = counts[b]
        
    return reconstruction

def run_part1(image):
    # encode the images
    qc = encoder(image)
    
    qobj = assemble(qc)
    #simulate the quantum circuit
    result = sim.run(qobj).result()

    counts = result.get_counts()
    new_image = decoder(counts)
    return qc,new_image

def run_part2(image):
    # Connect to simulator
    sim = Aer.get_backend('aer_simulator')
    
    #Create a composite circuit
    classifier = pickle.load(open("save.pickle", "rb"))
    encode = encoder(image)
    circuit = encode.compose(classifier)
    
    # Assemble, simulate and get histogram
    qobj = assemble(circuit)
    result = sim.run(qobj).result()
    counts = result.get_counts()
    
    # Get label
    label = histogram_to_category(counts)
    if label > 0.51:
        label = 1
    else:
        label = 0
    return circuit,label
############################
#      END YOUR CODE       #
############################

test()