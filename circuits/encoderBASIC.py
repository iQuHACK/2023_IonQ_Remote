import numpy as np
import pickle
import json
import os

import qiskit as qk
from qiskit import QuantumCircuit
from qiskit import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import quantum_info
from qiskit.execute_function import execute

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