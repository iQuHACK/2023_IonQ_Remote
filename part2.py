teamname="Feynman's Fashion Rescue"
task="part 2"

import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
import warnings

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import qiskit
from qiskit import *
from qiskit import quantum_info
from qiskit.execute_function import execute
from qiskit import BasicAer

from qiskit import Aer, execute
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit.circuit.library import TwoLocal, NLocal, RealAmplitudes, EfficientSU2
from qiskit.circuit.library import HGate, RXGate, RYGate, RZGate, CXGate, CRXGate, CRZGate
from qiskit_machine_learning.kernels import QuantumKernel

import numpy as np
import pickle
import json
import os
from collections import Counter
from sklearn.metrics import mean_squared_error
from typing import Dict, List

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
    """This function take a histogram representations of circuit execution results, and process into labels as described in 
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

#load the actual hackthon data (fashion-mnist)
images=np.load('data/images.npy')
labels=np.load('data/labels.npy')
#you can visualize it
import matplotlib.pyplot as plt
plt.imshow(images[1100])

#your submission

# Functions 'encode' and 'decode' are dummy.
def encode(image):
    print("Encoding...")
    number_of_qubits = math.ceil(math.log(28*28, 2))
    amplitude_vector = [0] * (2**number_of_qubits)
    sum_squares = 0
    amplitude_counter = 0
    for i in image:
        for j in i:
            sum_squares+=j**2
            amplitude_vector[amplitude_counter] = j
            amplitude_counter+=1
    global norm
    norm = 1/np.sqrt(sum_squares)
    amplitude_vector_new = [i*norm for i in amplitude_vector]
    
    # Some previous tests we were running -- ignore
    
    # global imtest
    # imtest=[[0]*28 for z in range(28)]
    # a_counter=0
    # for i in range(28):
    #     for j in range(28):
    #         imtest[i][j]=amplitude_vector[a_counter]#/norm
    #         a_counter+=1
            
            
    # print(amplitude_vector)
    qr = qiskit.QuantumRegister(number_of_qubits)
    qc = qiskit.QuantumCircuit(qr)
    qc.initialize(amplitude_vector_new, qr)

    print("Encoded!")
    return qc

def decode(histogram):
    print("Decoding...")      
    image = [[0] * 28 for z in range(28)]
    amplitude_counter=1
    for i in range(28):
        for j in range(28):
            image[i][j] = histogram.get(amplitude_counter, 0)#/norm
            amplitude_counter+=1
    print("Decoded!")
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
    # classifier=qiskit.QuantumCircuit.from_qasm_file('quantum_classifier.qasm')
    
    #encode image into circuit
    circuit = encode(image)
    image = decode(simulate(circuit))
    
    nsamples, nx, ny = image.shape
    images_reshaped = image.reshape((nsamples,nx*ny))
    
    
    
    
    #append with classifier circuit
    # nq1 = circuit.width()
    # nq2 = classifier.width()
    # nq = max(nq1, nq2)
    # qc = qiskit.QuantumCircuit(nq)
    # qc.append(circuit.to_instruction(), list(range(nq1)))
    # qc.append(classifier.to_instruction(), list(range(nq2)))
    
    
    
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
        
    return circuit,label


#score

#how we grade your submission

score=0
gatecount=0
n=len(dataset)

for data in dataset:
    #run part 2
    circuit,label=run_part2(data['image'])
    
    #count the gate used in the circuit for score calculation
    gatecount+=count_gates(circuit)[2]
    
    #check label
    if label==data['category']:
        score+=1
#score
score=score/n
gatecount=gatecount/n

print(score*(0.999**gatecount))

new_array=np.zeros((2000, 764))
img_counter = 0
pxl_counter = 0
for image in images:
    image_array = np.zeros((764))
    for i in image:
        for j in i:
            image_array[pxl_counter] = j
            pxl_counter += 1
        pxl_counter = 0
    new_array[img_counter] = image_array
    img_counter += 1


nsamples, nx, ny = images.shape
images_reshaped = images.reshape((nsamples,nx*ny))
# training quantum classifier
train_images, test_images, train_labels, test_labels = train_test_split(
    images_reshaped, labels, test_size=0.2, random_state=420)

sample_train, sample_val, labels_train, labels_val = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42)
fig = plt.figure()
LABELS = [0, 1]
num_labels = len(LABELS)
for i in range(num_labels):
    ax = fig.add_subplot(1, num_labels, i+1)
    img = train_images[train_labels==LABELS[i]][0].reshape((28, 28))
    ax.imshow(img, cmap="Greys")



standard_scaler = StandardScaler()
sample_train = standard_scaler.fit_transform(sample_train)
sample_val = standard_scaler.transform(sample_val)
test_images = standard_scaler.transform(test_images)

# Reduce dimensions
N_DIM = 5
pca = PCA(n_components=N_DIM)
sample_train = pca.fit_transform(sample_train)
sample_val = pca.transform(sample_val)
test_images = pca.transform(test_images)

# Normalize
min_max_scaler = MinMaxScaler((-1, 1))
sample_train = min_max_scaler.fit_transform(sample_train)
sample_val = min_max_scaler.transform(sample_val)
test_images = min_max_scaler.transform(test_images)


labels_train_0 = np.where(labels_train==0, 1, 0)
labels_val_0 = np.where(labels_val==0, 1, 0)

print(f'Original validation labels:      {labels_val}')
print(f'Validation labels for 0 vs Rest: {labels_val_0}')

pauli_map_0 = PauliFeatureMap(feature_dimension=N_DIM, reps=2, paulis = ['X', 'Y', 'ZZ'])
pauli_kernel_0 = QuantumKernel(feature_map=pauli_map_0, quantum_instance=Aer.get_backend('statevector_simulator'))

pauli_svc_0 = SVC(kernel='precomputed', probability=True)

matrix_train_0 = pauli_kernel_0.evaluate(x_vec=sample_train)
pauli_svc_0.fit(matrix_train_0, labels_train_0)

matrix_val_0 = pauli_kernel_0.evaluate(x_vec=sample_val, y_vec=sample_train)
pauli_score_0 = pauli_svc_0.score(matrix_val_0, labels_val_0)
print(f'Accuracy of discriminating between label 0 and others: {pauli_score_0*100}%')

matrix_test_0 = pauli_kernel_0.evaluate(x_vec=test_images, y_vec=sample_train)
pred_0 = pauli_svc_0.predict_proba(matrix_test_0)[:, 1]
print(f'Probability of label 0: {np.round(pred_0, 2)}')

pred = pauli_svc_0.predict_proba(matrix_test_0)[4]

classifier_file = open('quantum_classifier.pickle', 'wb')
pickle.dump(pauli_kernel_0, classifier_file)
classifier_file.close()
matrix_test_0 = pauli_kernel_0.evaluate(x_vec=images_reshaped)
pred = pauli_svc_0.predict_proba(images)[0]
print(pred)
