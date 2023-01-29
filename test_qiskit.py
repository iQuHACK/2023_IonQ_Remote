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
    images=np.load('./data'+'/images.npy')
    labels=np.load('./data'+'/labels.npy')
    
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
    # Initialize the quantum circuit for the image 
    # Pixel position
    idx = QuantumRegister(8, 'idx')
    # grayscale pixel intensity value
    intensity = QuantumRegister(8,'intensity')
    # classical register
    cr = ClassicalRegister(16, 'cr')
    # create the quantum circuit for the image
    qc_image = QuantumCircuit(intensity, idx, cr)
    # set the total number of qubits
    num_qubits = qc_image.num_qubits
    # Initialize the quantum circuit
    # Optional: Add Identity gates to the intensity values
    for idx in range(intensity.size):
        qc_image.i(idx)
    # Add Hadamard gates to the pixel positions    
    for i in range(8, 16, 1):
        #qc_image.h(8)
        qc_image.h(i)
    # Separate with barrier so it is easy to read later.
    qc_image.barrier()
    qc_image.draw()

    for i in range(64):
            theta = int ((out[i])*100) #Pixels from given image
            theta='{0:08b}'.format(theta)

            # Encode the second pixel whose value is (01100100):
            value01 = theta

            # Add the NOT gate to set the position at 01:
            qc_image.x(qc_image.num_qubits-1)

            # We'll reverse order the value so it is in the same order when measured.
            for idx, px_value in enumerate(value01[::-1]):
                if(px_value=='1'):
                    qc_image.ccx(num_qubits-1, num_qubits-2, idx)

            # Reset the NOT gate
            qc_image.x(num_qubits-1)

            qc_image.barrier()

    #qc_image.measure_all()
    #qc_image.draw()
    circuit=qc_image
    return circuit

def decode(histogram):
    
    # Define the histogramt
    hist=[]    
    values=[]
    values = list(counts_neqr.keys())
    type(values)
    key = [value.replace(" 0000000000000000", "") for value in values]
    pos=[k[:8] for k in key]
    pxl=[k[:8] for k in key]
    pxl = [int(p, 2) for p in pxl]
    print(pxl)

    def get_matrix(row_index, col_index, values):
        n = max(max(row_index), max(col_index)) + 1
        matrix = [[0 for _ in range(n)] for _ in range(n)]
        for r, c, v in zip(row_index, col_index, values):
        matrix[r][c] = v
    return matrix

    row_index = [0, 1, 2]
    col_index = [0, 1, 2]
    values = [1, 2, 3]
    print(get_matrix(row_index, col_index, values))

    from PIL import Image

    def array_to_image(arr):
        # Create a PIL image from the 2D array
        img = Image.fromarray(arr)
        # Convert the image to greyscale
        img = img.convert("L")
        return img

    # Create the histogram data
    histogram = pxl

    # Convert the histogram data to a grayscale image
    img = np.zeros((300, 300), dtype=np.uint8)
    for i, value in enumerate(histogram):
        img[:, i*30:(i+1)*30] = value
    image = img    
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
