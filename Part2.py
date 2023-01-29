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
from PIL import Image
from sklearn.decomposition import PCA


def run_part2(image):
    # load the quantum classifier circuit
    # a. Use encode(image) to convert the image into a quantum circuit
    qr = qiskit.QuantumRegister(8) # 8 qubits to represent the 8x8 image
    circuit = qiskit.QuantumCircuit(qr)
    # Add gates to the circuit to implement the classifier logic
    circuit.h(qr[0])
    circuit.cx(qr[0], qr[1])
    circuit.cx(qr[1], qr[2])

    # Save the circuit to a .pickle file
    with open("quantum_classifier.pickle", "wb") as file:
        pickle.dump(circuit, file)

    
    en_image = encode(image)
    
    # b. Append the circuit with the classifier circuit loaded from the .pickle file
    with open("quantum_classifier_5qbits.pickle", "rb") as file:
        classifier_circuit = pickle.load(file)
        
    #circuit = encoded_image + classifier_circuit
    circuit = en_image.compose(classifier_circuit)
    
    # c. Simulate the circuit (encoded_image+classifier_circuit) and get a histogram
    backend = qiskit.BasicAer.get_backend("qasm_simulator")
    result = qiskit.execute(circuit, backend, shots=1024).result()
    ob1 = assemble(circuit)
    res = backend.run(ob1).result()
    
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

