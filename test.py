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
    for i in range(3,20): # only 1- and 2-qubit gates
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
import math
from scipy.ndimage import zoom

# Image properties
SIZE = 7 # Image width
NB_PX_IMG = SIZE ** 2

# quantum parameters
N = math.ceil(math.log2(SIZE))
NB_QUBITS = 2 * N + 1
NB_PX = 2 ** (2 * N)



def pixel_value_to_theta(pixel: float) -> float:
    return pixel / 255 * (np.pi / 2)

def theta_to_pixel_value(theta: float) -> int:
    return int(theta / (np.pi / 2) * 255)

def get_proba(counts: dict) -> dict:
    sums = sum(map(lambda x: x[1], counts.items()))
    return {key: value / sums for key, value in counts.items()}


def recursive_ry(circuit, theta, mask):
    if mask.sum() == 2:
        idxs = np.where(mask == 1)[0]
        circuit.cry(theta/2, idxs[0], NB_QUBITS - 1)
        circuit.cx(idxs[0], idxs[1])
        circuit.cry(-theta/2, idxs[1], NB_QUBITS - 1)
        circuit.cx(idxs[0], idxs[1])
        circuit.cry(theta/2, idxs[1], NB_QUBITS - 1)
    else:
        idx1, idx2 = np.where(mask == 1)[0][:2]
        
        maskn = mask.copy()
        maskn[idx2] = 0
        recursive_ry(circuit, theta/2, maskn)
        # c3ry = RYGate(theta).control(NB_QUBITS - 1)
        # circuit.append(c3ry, ry_qbits)
        circuit.cx(idx1, idx2)

        maskn = mask.copy()
        maskn[idx1] = 0
        recursive_ry(circuit, -theta/2, maskn)
        circuit.cx(idx1, idx2)

        maskn = mask.copy()
        maskn[idx1] = 0
        recursive_ry(circuit, theta/2, maskn)




def encode(image):
    circuit = qiskit.QuantumCircuit(NB_QUBITS)
    image = image[::4, ::4]
    # Get the theta values for each pixel
    image = image.flatten()
    thetas = [pixel_value_to_theta(pixel) for pixel in image]
    thetas += [0] * (NB_PX - NB_PX_IMG)

    # Apply Hadamard gates for all qubits except the last one
    for i in range(NB_QUBITS - 1):
        circuit.h(i)

    switches = [bin(0)[2:].zfill(NB_QUBITS - 1)] + [
        bin(i ^ (i - 1))[2:].zfill(NB_QUBITS - 1) for i in range(1, NB_PX)
    ]


    # Apply the rotation gates
    prev_switch = switches[0]
    for i in range(NB_PX):
        theta = thetas[i] # pixel_value_to_theta(intensity_count_expression[i][0])
        
        # do not do zero rotation
        if theta != 0:
            switch = np.binary_repr(i, NB_QUBITS - 1)

            # Apply x gate to the i-th qubit if the i-th bit of the switch is 1
            for j in range(NB_QUBITS - 1):
                if switch[j] != prev_switch[j]:
                    circuit.x(j)
                # if switch[j] == "1":
                #     circuit.x(j - 1)
            prev_switch = switch

            recursive_ry(circuit, 2*theta, np.array([1]*(NB_QUBITS - 1)))

            #circuit.barrier()

    # check all qbits are at 1 for the measurements
    for j in range(NB_QUBITS - 1):
        if prev_switch[j] != "1":
            circuit.x(j)

    circuit.measure_all()
    
    from qiskit.transpiler.passes import RemoveBarriers
    circuit = RemoveBarriers()(circuit)
    
    return circuit

def decode(counts: dict) -> np.ndarray:
    histogram = get_proba(counts)
    img = np.zeros(NB_PX)  # we have a square image

    for i in range(NB_PX):
        bin_str: str = np.binary_repr(i, width=NB_QUBITS - 1)

        cos_str = "0" + bin_str[::-1]
        sin_str = "1" + bin_str[::-1]

        theta = 0
        if cos_str in histogram:
            prob_cos = histogram[cos_str]
            theta = math.acos(np.clip(2**N * math.sqrt(prob_cos), 0, 1))
        else:
            prob_cos = 0

        # not needed?
        if sin_str in histogram:
            prob_sin = histogram[sin_str]
            theta = math.asin(np.clip(2**N * math.sqrt(prob_sin), 0, 1))
        else:
            prob_sin = 0

        img[i] = theta_to_pixel_value(theta)

    img = img[:NB_PX_IMG]
    img = img.reshape(SIZE, SIZE)
    return zoom(img, 4, order=0)

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
    classifier=qiskit.QuantumCircuit.from_qasm_file('part2.qasm')
    
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