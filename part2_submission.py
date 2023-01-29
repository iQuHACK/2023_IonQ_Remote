teamname = "Hilbert's Qerudites"
task = 'part 2'

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

q = qiskit.QuantumRegister(14)
classifier_qiskit = qiskit.QuantumCircuit(q)
for i in range(14):
    classifier_qiskit.rx(0.5,i)

file_pi = open('quantum_classifier.pickle', 'wb') 
pickle.dump(classifier_qiskit, file_pi)

def run_part2(image):

    #loade the quantum classifier circuit
    # we have attempted to make our feature map such that it can itself be used for classification,
    # so we have just added this naive classifier circuit
    
    with open('quantum_classifier.pickle', 'rb') as f:
        classifier=pickle.load(f)
    
    #encode image into circuit,encode takes image as a 28x28 numpy array
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
        
    return circuit,label
#score