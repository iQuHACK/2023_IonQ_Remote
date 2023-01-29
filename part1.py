#submission to part 1, you should make this into a .py file
import qiskit
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
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.circuit.library.standard_gates import XGate
from qiskit import transpile

images=np.load('data/images.npy')
labels=np.load('data/labels.npy')

n=len(images)
mse=0
gatecount=0

def max_pooling(image):
    image=np.reshape(image,(28,28))
    new_image=np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            new_image[i,j]=int(np.max(image[i*4:(i+1)*4,j*4:(j+1)*4])*10000)
    return new_image


#max depooling (8x8 -> 28x28)
def max_depooling(image):
    image=np.reshape(image,(7,7))
    new_image=np.zeros((28,28))
    for i in range(7):
        for j in range(7):
            new_image[i*4:(i+1)*4,j*4:(j+1)*4]=image[i,j]/10000
    return new_image

custom_gate = QuantumCircuit(1, name='custom_gate')
custom_gate.x(0)
custom_gate_gate = custom_gate.to_gate().control(6)

# Functions 'encode' and 'decode' are dummy.
def encode(image):
    maxPooledImg=max_pooling(image)
    idx = QuantumRegister(3, 'idx')
    idy = QuantumRegister(3, 'idy')
    # grayscale pixel intensity value
    intensity = QuantumRegister(8,'intensity')
    # classical register
    cr = ClassicalRegister(14, 'cr')

    # create the quantum circuit for the image
    qc_image = QuantumCircuit(intensity, idx, idy, cr)

    # set the total number of qubits
    num_qubits = qc_image.num_qubits

    # initialize the qubits
    qc_image.h(range(8,14))
    qc_image.barrier()
    
    for i in range(7):
        for j in range(7):
            for ix,v in enumerate(f'{int(i):b}'.zfill(3)[::-1]):
                if v=='1':
                    qc_image.x([ix+8])
            for iy,v in enumerate(f'{int(j):b}'.zfill(3)[::-1]):
                if v=='1':
                    qc_image.x([iy+11])
            sint=f'{int(maxPooledImg[i,j]):b}'.zfill(8)
            #print(sint)
            for idx, px_value in enumerate(sint[::-1]):
                if px_value=='1':
                    qc_image.append(custom_gate_gate,[8,9,10,11,12,13,idx])
            for ix,v in enumerate(f'{int(i):b}'.zfill(3)[::-1]):
                if v=='1':
                    qc_image.x([ix+8])
            for iy,v in enumerate(f'{int(j):b}'.zfill(3)[::-1]):
                if v=='1':
                    qc_image.x([iy+11])
            qc_image.barrier()
      
    
    qc_image.draw()
    #transpile the circuit
    qc_image=transpile(qc_image,basis_gates=['cx','x','y','z','id'])
    return qc_image

def decode(histogram):
    # decode the histogram into an image where the keys are the qubit values
    # the first 3 qubits are for the y coordinate
    # the second 3 qubits are for the x coordinate
    # the last 8 qubits are for the grayscale pixel intensity value
    
    print(histogram)
    keys=histogram.keys()
    img=np.zeros((7,7))
    for i in keys:
        si=f'{int(i):b}'.zfill(14)
        print(si)
        x=int(si[8:11],2) if int(si[8:11],2)<7 else 0
        y=int(si[11:14],2) if int(si[11:14],2)<7 else 0
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

    return circuit,image_re