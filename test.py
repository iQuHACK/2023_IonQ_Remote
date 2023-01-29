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
# General imports
import os
import gzip
import numpy as np
import matplotlib.pyplot as plt
from pylab import cm
import warnings

warnings.filterwarnings("ignore")

# scikit-learn imports
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Qiskit imports
from qiskit import Aer, execute, BasicAer
from qiskit.utils import QuantumInstance
from qiskit.circuit import QuantumCircuit, Parameter, ParameterVector
from qiskit.circuit.library import PauliFeatureMap, ZFeatureMap, ZZFeatureMap
from qiskit.circuit.library import TwoLocal, NLocal, RealAmplitudes, EfficientSU2
from qiskit.circuit.library import HGate, RXGate, RYGate, RZGate, CXGate, CRXGate, CRZGate
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit_machine_learning.algorithms.classifiers import QSVC
import pickle

if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    data_path = '.'

#define utility functions

def simulate(qc_image):
    qc_image.measure(range(6),range(6))
    aer_sim = Aer.get_backend('aer_simulator')
    t_qc_image = transpile(qc_image, aer_sim)
    qobj = assemble(t_qc_image, shots=1000)
    job_neqr = aer_sim.run(qobj)
    result_neqr = job_neqr.result()
    counts_neqr = result_neqr.get_counts()
    return counts_neqr


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
def encode(image):
     image = image/255

# Reshape the image
    image = image.reshape(image.shape[0], *(28,1))
    image = tf.image.resize(image, (2,2)).numpy()

    idx = QuantumRegister(2, 'idx')
# grayscale pixel intensity value
    intensity = QuantumRegister(4,'intensity')
# classical register
    cr = ClassicalRegister(6, 'cr')

# create the quantum circuit for the image
    qc_image = QuantumCircuit(intensity, idx, cr)

# set the total number of qubits
    num_qubits = qc_image.num_qubits
    for idx in range(intensity.size):
        qc_image.i(idx)

# Add Hadamard gates to the pixel positions    
    qc_image.h(4)
    qc_image.h(5)

# Separate with barrier so it is easy to read later.
    qc_image.barrier()
    bin_val = binary_encode(image)
    # Use join() to concatenate the elements of the array into a string
    result_string = ''.join(map(str, bin_val))
    value01 = result_string

# Add the NOT gate to set the position at 01:
    qc_image.x(qc_image.num_qubits-1)

# We'll reverse order the value so it is in the same order when measured.
    for idx, px_value in enumerate(value01[::-1]):
        if(px_value=='1'):
            qc_image.ccx(num_qubits-1, num_qubits-2, idx)

# Reset the NOT gate
    qc_image.x(num_qubits-1)

    qc_image.barrier()
    return qc_image

def decode(histogram):
    hist_val = list(histogram.keys())
    binary_values = hist_val
    # reshape to 2x2 array
    binary_values = np.array(binary_values, dtype=np.uint8)
    image = binary_values.reshape((2, 2))
    plt.imshow(image)
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
    pauli_map = PauliFeatureMap(feature_dimension=5, reps=2, paulis = ['X', 'Y', 'ZZ'])
    pauli_kernel = QuantumKernel(feature_map=pauli_map, quantum_instance=Aer.get_backend('statevector_simulator'))
    matrix_train = pauli_kernel.evaluate(x_vec=sample_train)
    qsvm = QSVC(quantum_kernel=pauli_kernel)
    qsvm.fit(sample_train, labels_train)
    backend = BasicAer.get_backend('qasm_simulator')
    quantum_instance = QuantumInstance(backend, shots=1024)# seed_simulator=seed, seed_transpiler=seed)
    pred=qsvm.predict(sample_val)
    print(qsvm.score(sample_val,labels_val))
    with open("quantum_classifier.pickle", "wb") as file:
        pickle.dump(pauli_map, file)
    with open("quantum_classifier.pickle", "rb") as file:
        classifier_circuit = pickle.load(file)    
    return classifier_circuit,qsvm.predict(image)

############################
#      END YOUR CODE       #
############################

test()
