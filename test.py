import qiskit
from qiskit import quantum_info
from qiskit.execute_function import execute
from qiskit import BasicAer
import numpy as np
import pickle
import json
import os
import math
import sys
from collections import Counter
from sklearn.metrics import mean_squared_error
from typing import Dict, List
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
    for i in range(2,20):
        assert counter[i]==0
        
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
def encode(image):
    # initialize the circuit
    row = qiskit.QuantumRegister(4, "row")
    col = qiskit.QuantumRegister(4, "col")
    val = qiskit.QuantumRegister(1, "val")
    #cr = qiskit.QuantumRegister(3, "cr")
    qc = qiskit.QuantumCircuit(row, col, val)

    # assuming we have 28 x 28 image, first pad it to 32 x 32
    img = np.pad(image, 2)
    
    # then downscale it to 16 x 16
    x, y = img.shape
    new_img = np.zeros((x//2, y//2))
    
    # simple average pooling
    for i in range(x//2):
        for j in range(y//2):
            new_img[i, j] = (img[i*2, j*2] + img[i*2+1, j*2] + img[i*2, j*2+1] + img[i*2+1, j*2+1]) / 4
    
    # finally, bin the pixel intensities into 8 values
    new_img *= 255**2
    new_img[new_img<30] = 0
    new_img[new_img>=230] = 0.875
    new_img[new_img>=200] = 0.75
    new_img[new_img>=165] = 0.625
    new_img[new_img>=130] = 0.5
    new_img[new_img>=100] = 0.375
    new_img[new_img>=65] = 0.25
    new_img[new_img>=30] = 0.125
    new_img = (new_img*8).astype("uint8")
    
    # use FRQI for encoding this image, using 9 total qubits:
    # 4 qubits for row, 4 for column, 1 for intensity
    
    # add Hadamard gates for input bits first
    for i in range(8):
        qc.h(i)
    qc.barrier()
    
    # now set gates for intensity value of each pixel
    for i in range(4):
        for c1 in range(2):
            for j in range(4):
                for c2 in range(2):
                    theta = new_img[i*(c1+1), j*(c2+1)]*math.pi/8
                    r, c = i, j+4

                    qc.cry(theta, r, 8)
                    qc.cx(r, c)
                    qc.cry(-theta, c, 8)
                    qc.cx(r, c)
                    qc.cry(theta, c, 8)
                    qc.barrier()
                    qc.x(c)
                qc.x(r)
    return qc

def decode(histogram):
    r, c = 16, 16
    # convert numbers to binary form
    bin_hist = {f"{k:09b}": v for k, v in histogram.items()}
    
    # create an array for storing image
    img = np.zeros((r, c))
    
    # get intensity at each pixel
    for a in range(256):
        i, j = a//r, a%c
        try:
            # I assume the ratio of |0> and |1> states for each pixel
            # value is the same as the CRy angle theta
            ones = bin_hist.get(f"1{a:08b}", 0)
            zeros = bin_hist.get(f"0{a:08b}", 0)
            img[i, j] = ones / (ones+zeros)
        except KeyError:
            img[i, j] = 0
    #img = img.astype("uint8")
    
    # upscale 16x16 to 32x32
    new_img = np.zeros((r*2, c*2))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            val = img[i, j]
            new_img[i*2, j*2] = new_img[i*2+1, j*2] = new_img[i*2, j*2+1] = new_img[i*2+1, j*2+1] = val
    
    # and remove padding
    new_img = new_img[2:-2, 2:-2]
    return new_img

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
