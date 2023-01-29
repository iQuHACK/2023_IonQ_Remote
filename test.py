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

if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    data_path = 'data'

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
    
    #     print(i)
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
    
    # score=0
    # gatecount=0
    # n=len(images)

    # for i in range(n):
    #     #run part 2
    #     circuit,label=run_part2(images[i])

    #     #count the gate used in the circuit for score calculation
    #     gatecount+=count_gates(circuit)[2]

    #     #check label
    #     if label==labels[i]:
    #         score+=1
    # #score
    # score=score/n
    # gatecount=gatecount/n

    # score_part2=score*(0.999**gatecount)
    
    print(score_part1)


############################
#      YOUR CODE HERE      #
############################
import math
from qiskit import QuantumCircuit
import numpy as np


def decode(hist):
    def bin_rep(x, n=8):
        t = "{0:b}".format(x)
        if len(t) < n:
            t = '0'*(n-len(t)) + t
        elif len(t) > n:
            t = t[len(t)-n:]
        return t
    
    n = 16
    img = np.zeros((n,n))
    
    for i in range(n):
        for j in range(n):
            st =  bin_rep(i,4) + bin_rep(j,4) 
            st = st[::-1]
            st0 = '0' + st
            st1 = '1' + st
            if st0 in hist and st1 in hist:
                c0, c1 = hist[st0], hist[st1]
                t = c0+c1
                c0,c1 = c0/t, c1/t
                img[n-i-1][j] = math.acos(c0-c1)
            else:
                img[n-i-1][j] = 0
            
    img = img * 2 * 240 / np.pi
    img = img[:-2,2:]
    for i in range(14):
        img[i] = np.flip(img[i])
        
    return double(img)

def encode(data):
    data = data * 255 / data.max()
    data = data.astype(int)
    data = pooling(data)
    data = data.astype(int)
    n = 8
    q = 1
    ac = 7
    num_qubits = n+q+ac #8 qubits for pixels and 6 qubits for data 
    qc_image = QuantumCircuit(num_qubits, n+q) 

    for i in range(n):
        qc_image.h(i)


    # Add the CNOT gates 
    for idx , px in np.ndenumerate(data):
        if px > 15: 
            qc_image = apply_x(qc_image,*idx)
            qc_image = apply_tofolli(qc_image,n,n+q,n, np.pi * px / (2*240))
            qc_image = apply_x(qc_image,*idx)

    # #run circuit in backend and get the state vector 
    # backend = BasicAer.get_backend('statevector_simulator')
    # result = execute(qc_image, backend=backend).result() 
    # output = result.get_statevector(qc_image) 

    for i in range(n+q):
        qc_image.measure(i,i)

    return qc_image

def bin_rep(x, n=8):
    t = "{0:b}".format(x)
    if len(t) < n:
        t = '0'*(n-len(t)) + t
    elif len(t) > n:
        t = t[len(t)-n:]
    return t

def tofolli(qc: QuantumCircuit,x,y,t):
    qc.barrier()
    qc.h(t)
    qc.cx(y,t)
    qc.tdg(t)
    qc.cx(x,t)
    qc.t(t)
    qc.cx(y,t)
    qc.tdg(t)
    qc.cx(x,t)
    qc.t(y)
    qc.t(t)
    qc.cx(x,y)
    qc.h(t)
    qc.t(x)
    qc.tdg(y)
    qc.cx(x,y)
    qc.barrier()
    return qc

def apply_tofolli(qc, n, anc, target, theta):
    anc_st = anc
    qc.barrier()
    for i in range(n):
        if i == 0:
            qc = tofolli(qc,i,i+1,anc)
        elif i == 1:
            ...
        else:
            qc = tofolli(qc,i, anc_st, anc_st+1)
            anc_st += 1

    qc.cry(theta, anc_st, target)
    anc_st -= 1

    for i in reversed(range(n)):
        if i == 0:
            qc = tofolli(qc,i,i+1,anc)
        elif i == 1:
            pass
        else:
            qc = tofolli(qc,i, anc_st, anc_st+1)
            anc_st -= 1
    
    qc.barrier()

    return qc

def apply_x(qc, x,y):
    x = bin_rep(x,4)
    y = bin_rep(y,4)
    t = x+y
    qc.barrier()
    for i, v in enumerate(t):
        if v == '1':
            qc.x(i)
    qc.barrier()
    return qc

def pooling(mat,ksize=(2,2),pad=False):
    m, n = mat.shape[:2]
    ky,kx=ksize

    _ceil=lambda x,y: int(np.ceil(x/float(y)))

    if pad:
        ny=_ceil(m,ky)
        nx=_ceil(n,kx)
        size=(ny*ky, nx*kx)+mat.shape[2:]
        mat_pad=np.full(size,np.nan)
        mat_pad[:m,:n,...]=mat
    else:
        ny=m//ky
        nx=n//kx
        mat_pad=mat[:ny*ky, :nx*kx, ...]

    new_shape=(ny,ky,nx,kx)+mat.shape[2:]

    result=np.nanmean(mat_pad.reshape(new_shape),axis=(1,3))

    return result

def double(img):
    new_img = np.zeros((28,28))
    x,y = 0,0
    for (i,j),v in np.ndenumerate(img):
        new_img[2*i, 2*j] = v
        new_img[2*i+1, 2*j] = v
        new_img[2*i, 2*j+1] = v
        new_img[2*i+1, 2*j+1] = v
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