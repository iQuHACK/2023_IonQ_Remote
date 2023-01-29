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
# import cv2
from skimage.transform import resize

if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    data_path = './data/'


# define utility functions
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
    # for i in range(2,20):
    #     assert counter[i]==0
        
    return counter


def image_mse(image1,image2):
    # Using sklearns mean squared error:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    return mean_squared_error(255*image1,255*image2)

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
# we add preprocess data for our model
def amplitude_encode(img_data):
    # Calculate the RMS value
    rms = np.sqrt(np.sum(np.sum(img_data ** 2, axis=1)))

    # Create normalized image
    image_norm = []
    for arr in img_data:
        for ele in arr:
            image_norm.append(ele / rms)

    # Return the normalized image as a numpy array
    return np.array(image_norm)


def encode(image):
    # NOTE: can actually resize the image to make it larger i.e. use n = 32
    n = 16

    im = resize(image, output_shape=(n, n))

    image_norm = amplitude_encode(im)
    data_qb = 8  # math.log2(n*n)
    anc_qb = 1
    total_qb = data_qb + anc_qb

    # Initialize the amplitude permutation unitary
    D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)

    qc = qiskit.QuantumCircuit(total_qb)
    qc.initialize(image_norm, range(1, total_qb))
    qc.h(0)
    qc.unitary(D2n_1, range(total_qb))
    qc.h(0)
    # display(qc.draw('mpl', fold=-1))
    
    return qc


def decode(histogram):
    n = 16
    data_qb = 8  # math.log2(n*n)
    anc_qb = 1
    total_qb = data_qb + anc_qb

    sva = np.zeros(2 ** total_qb)
    for key, value in histogram.items():
        sva[key] = value

    # NOTE: not sure about subsampling every second item
    return resize(sva[::2].reshape((n, n)), output_shape=(28,28))


def run_part1(image):
    # encode image into a circuit
    circuit = encode(image)

    # simulate circuit
    histogram = simulate(circuit)

    # reconstruct the image
    image_re = decode(histogram)

    return circuit, image_re


def run_part2(image):
    # load the quantum classifier circuit
    classifier = qiskit.QuantumCircuit.from_qasm_file('quantum_classifier.qasm')

    # encode image into circuit
    circuit = encode(image)

    # append with classifier circuit
    nq1 = circuit.width()
    nq2 = classifier.width()
    nq = max(nq1, nq2)
    qc = qiskit.QuantumCircuit(nq)
    qc.append(circuit.to_instruction(), list(range(nq1)))
    qc.append(classifier.to_instruction(), list(range(nq2)))

    # simulate circuit
    histogram = simulate(qc)

    # convert histogram to category
    label = histogram_to_category(histogram)

    # thresholding the label, any way you want
    if label > 0.45:
        label = 1
    else:
        label = 0

    return circuit, label


############################
#      END YOUR CODE       #
############################

test()
