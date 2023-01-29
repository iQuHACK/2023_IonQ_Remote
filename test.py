import qiskit
from qiskit import quantum_info, QuantumCircuit
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
from sklearn.decomposition import PCA
import tqdm


import matplotlib.pyplot as plt

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
    for i in range(3,20):
        assert counter[i]==0
        
    return counter


def image_mse(image1,image2):
    # Using sklearns mean squared error:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    return mean_squared_error(255*image1,255*image2)

# [normalize]
def normalize(row):
    #We calculate the squareroot of the sum of the square values of the row
    normalization_factor = np.sqrt(np.sum(row**2)) 
    if normalization_factor == 0.:
        #If the sum is zero we return a 0
        return 0.0
    #Else we divide each value between the sum value above
    row = row / normalization_factor
    return row, normalization_factor

def test():
    #load the actual hackthon data (fashion-mnist)
    images=np.load(data_path+'/images.npy')
    labels=np.load(data_path+'/labels.npy')
    
    #test part 1
    global pca, AMPLITUDE_ENCODING_N_QUBITS,IMAGE_SIZE, dataset

    AMPLITUDE_ENCODING_N_QUBITS = 4
    IMAGE_SIZE = 28
    N_IMAGES = len(images)

    pca = PCA(n_components=2**AMPLITUDE_ENCODING_N_QUBITS)
    
    data = images.reshape(N_IMAGES, IMAGE_SIZE * IMAGE_SIZE)
    pca.fit(data)
    
    # Apply dimensionality reduction on your data
    images_pca = pca.transform(data)
    
    dataset = []
    for i in tqdm.tqdm(range(len(images))):
        image_pca = images_pca[i]
        image_pca_min_ = image_pca.min()
        image_pca_positive = image_pca - image_pca_min_
        
        dataset_i = {}
        
        normalized_state, normalization_factor = normalize(image_pca_positive)
        
        dataset_i["image"] = images[i]
        dataset_i["image_vector"] = normalized_state
        dataset_i["label"] = labels[i]
        dataset_i["normalization_factor"] = normalization_factor
        dataset_i["pca_min_"] = image_pca_min_
        
        dataset.append(dataset_i)
    
    n=len(images)
    mse=0
    gatecount=0
    
    for i in tqdm.tqdm(range(n)):
        #encode image into circuit
        circuit,image_re=run_part1(dataset[i]["image_vector"])
        image_re = np.asarray(image_re)

        #count the number of 2qubit gates used
        gatecount += count_gates(circuit)[2]

        #calculate mse
        mse+=image_mse(dataset[i]["image"], image_re)

    #fidelity of reconstruction
    f=1-mse/n
    gatecount=gatecount/n

    #score for part1
    score_part1=f*(0.999**gatecount)
    
    print(score_part1)

    #test part 2
    
    score=0
    gatecount=0
    n=len(images)

    for i in tqdm.tqdm(range(n)):
        #run part 2
        circuit,label=run_part2(dataset[i]["image_vector"])

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
    qc = QuantumCircuit(AMPLITUDE_ENCODING_N_QUBITS)

    qc.initialize(image)
    
    for i in range(AMPLITUDE_ENCODING_N_QUBITS + 2):
        qc = qc.decompose()
        
    return qc

def decode(histogram):
    histogram_array = np.zeros(2 ** AMPLITUDE_ENCODING_N_QUBITS)
    for i in range(2 ** AMPLITUDE_ENCODING_N_QUBITS):
        histogram_array[i] = histogram.get(i, 0)
        
    root = np.sqrt(histogram_array)
    
    root = root * dataset[i]["normalization_factor"]
    
    root = root + dataset[i]["pca_min_"]

    reconstruction = pca.inverse_transform([root])

    
    image = reconstruction.reshape(IMAGE_SIZE, IMAGE_SIZE)
    
    return image

def run_part1(image):
    #encode image into a circuit
    circuit=encode(image)

    #simulate circuit
    histogram=simulate(circuit)

    #reconstruct the image
    image_re=decode(histogram)

    return circuit,image_re

def run_part2(image):
    # load the quantum classifier circuit
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
    
    #thresholding the label, any way you want
    if label>0.5:
        label=1
    else:
        label=0
        
    return circuit,label

############################
#      END YOUR CODE       #
############################

test()