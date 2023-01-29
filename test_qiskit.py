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
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.standard_gates import XGate
from qiskit import transpile

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
    inusfd=0

    for image in images:
        #encode image into circuit
        circuit,image_re=run_part1(image)
        image_re = np.asarray(image_re)
        print(inusfd)
        inusfd+=1

        #count the number of 2qubit gates used
        gatecount+=count_gates(circuit)[2]

        #calculate mse
        mse+=image_mse(image,image_re)

    #fidelity of reconstruction
    f=1-mse/n
    gatecount=gatecount/n

    #score for part1
    score_part1=f*(0.999**gatecount)
    print("score_part1:",score_part1)
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
def max_pooling(image):
    image=np.reshape(image,(28,28))
    new_image=np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            new_image[i,j]=int(np.mean(image[i*7:(i+1)*7,j*7:(j+1)*7])*10000)
    return new_image

#max depooling (4x4 -> 28x28)
def max_depooling(image):
    image=np.reshape(image,(4,4))
    new_image=np.zeros((28,28))
    for i in range(4):
        for j in range(4):
            # add gradient here
            new_image[i*7:(i+1)*7,j*7:(j+1)*7]=image[i,j]/10000
    return new_image
custom_gate = QuantumCircuit(1, name='custom_gate')
custom_gate.x(0)
custom_gate_gate = custom_gate.to_gate().control(4)

# Functions 'encode' and 'decode' are dummy.
def encode(image):
    maxPooledImg=max_pooling(image)
    idx = QuantumRegister(2, 'idx')
    idy = QuantumRegister(2, 'idy')
    # grayscale pixel intensity value
    intensity = QuantumRegister(6,'intensity')
    # classical register
    cr = ClassicalRegister(10, 'cr')

    # create the quantum circuit for the image
    qc_image = QuantumCircuit(intensity, idx, idy, cr)

    # set the total number of qubits
    num_qubits = qc_image.num_qubits

    # initialize the qubits
    qc_image.h(range(6,10))
    qc_image.barrier()
    
    for i in range(4):
        for j in range(4):
            for ix,v in enumerate(f'{int(i):b}'.zfill(2)[::-1]):
                if v=='1':
                    qc_image.x([ix+6])
            for iy,v in enumerate(f'{int(j):b}'.zfill(2)[::-1]):
                if v=='1':
                    qc_image.x([iy+8])
            sint=f'{int(maxPooledImg[i,j]):b}'.zfill(6)
            #print(sint)
            for idx, px_value in enumerate(sint[::-1]):
                if px_value=='1':
                    qc_image.append(custom_gate_gate,[6,7,8,9,idx])
            for ix,v in enumerate(f'{int(i):b}'.zfill(2)[::-1]):
                if v=='1':
                    qc_image.x([ix+6])
            for iy,v in enumerate(f'{int(j):b}'.zfill(2)[::-1]):
                if v=='1':
                    qc_image.x([iy+8])
            qc_image.barrier()
      
    
    qc_image.draw()
    #transpile the circuit
    qc_image=transpile(qc_image,basis_gates=['cx','u1','u2','u3','id'])
    return qc_image

def decode(histogram):
    # decode the histogram into an image where the keys are the qubit values
    # the first 3 qubits are for the y coordinate
    # the second 3 qubits are for the x coordinate
    # the last 8 qubits are for the grayscale pixel intensity value
    
    print(histogram)
    keys=histogram.keys()
    img=np.zeros((4,4))
    for i in keys:
        si=f'{int(i):b}'.zfill(10)
        print(si)
        x=int(si[6:8],2) if int(si[6:8],2)<4 else 0
        y=int(si[8:10],2) if int(si[8:10],2)<4 else 0
        img[x,y]=int(si[0:8],2)
        image=max_depooling(img)
    return image

def run_part1(image):
    #encode image into a circuit
    circuit=encode(image)

    #simulate circuit
    histogram=simulate(circuit)

    #reconstruct the image
    image_re=decode(histogram)
    
    


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
