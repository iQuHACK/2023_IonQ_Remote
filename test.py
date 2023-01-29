import qiskit
from qiskit import quantum_info, QuantumCircuit
from itertools import chain
from qiskit.execute_function import execute
from qiskit import BasicAer
from qiskit.circuit.library import ZZFeatureMap
import numpy as np
import pickle
import json
import os
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
# GLOBAL VARIABLES
DIM = 28
NUM_QUBITS = 10

def encode_img(image, register_num):
    ''' encoding of image using QPIE method '''
    img = list(chain(*image))
    pix_val = img

    # normalize
    pix_norm = np.linalg.norm(pix_val)
    pix_val = np.array(pix_val)
    arr_norm = pix_val/pix_norm
    arr_norm = arr_norm.tolist()

    # Encode onto the quantum register
    qc = QuantumCircuit(register_num)
    # test = arr_norm.append(np.zeros(2**10-arr_norm.shape))
    test = arr_norm + np.zeros(2**register_num-DIM**2).tolist()
    qc.initialize(test, qc.qubits)
    return qc


def encode(image):
    ''' final wrapper function (for submission) '''
    return encode_img(255*255*image, register_num=NUM_QUBITS)


def decode_img(histogram):
    ''' decoding (written by prathu) '''
    pixelnums = list(range(DIM**2))
    for pix in pixelnums:
        if pix not in histogram.keys():
            # grayscale pixel value is 0
            histogram.update({pix: 0})

    histnew = dict(sorted(histogram.items()))

    histdata = []
    # for i in enumerate(histnew):
    for i in range(len(histnew)):
        histdata.append(histnew[i])
    histdata = np.array(histdata)
    histarr = np.reshape(histdata, (DIM, DIM))

    return histarr


def decode(histogram):
    ''' final wrapper function (for submission) '''
    return decode_img(histogram)

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
    # classifier=qiskit.QuantumCircuit.from_qasm_file('quantum_classifier.qasm')

    with open('quantum_classifier_500samples.pkl','rb') as f:
        contents = pickle.load(f)
    
    classifier = contents['qsvm']#for classical feature maps
    pca = contents['pca'] #for classical feature maps

    #encode image into circuit
    img = pca.transform(image.reshape(-1, DIM*DIM)) #for classical feature maps

    circuit=encode(image) 
    
    #append with classifier circuit
    nq1 = circuit.width() #uncomment for quantum feature map
    nq2 = classifier.width() #uncomment for quantum feature map
    # nq = max(nq1, nq2) #uncomment for quantum feature map
    qc = qiskit.QuantumCircuit(nq2) #uncomment for quantum feature map and make nq2 to nq
    # qc.append(circuit.to_instruction(), list(range(nq1))) #uncomment for quantum feature map
    qc.append(classifier.to_instruction(), list(range(nq2))) #uncomment for quantum feature map
    
    #simulate circuit
    # histogram=simulate(qc) #uncomment for quantum feature map

    label = classifier.predict(img)[0] #for classical feature maps
    #convert histogram to category
    # label=histogram_to_category(histogram) #uncomment for quantum feature map
    
    #thresholding the label, any way you want
    #uncomment for quantum feature map
    # if label>0.5:
    #     label=1
    # else:
    #     label=0
        
    return circuit,label

############################
#      END YOUR CODE       #
############################

test()