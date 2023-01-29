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
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.decomposition import TruncatedSVD
    from sklearn.manifold import TSNE

    # The images are somehow 28 by 27 while the Kaggle says it is 28x28.
    #     I don't know why they lost a row or column.
    width = 28
    length = 27
    data_path = "./dataset/"

    # Loading train data
    train_data = np.load("/content/drive/MyDrive/2023_IonQ_Remote/data/images.npy")

    # Showing an image
    image = train_data[1, 1:].reshape((width, length))
    plt.imshow(image)
    plt.show()

    #Extracting features and labels from the dataset and truncating the dataset to 10,000 datapoints
    train_data_features = train_data[:10000, 1:]
    train_data_labels = train_data[:10000, :1].reshape(-1,)
    train_data_features = train_data_features.reshape(train_data_features.shape[0], -1)


    # Using SVD to reduce dimensions to 10
    tsvd = TruncatedSVD(n_components=2)
    X_SVD = tsvd.fit_transform(train_data_features)

    # Use t-SNE technique to reduce dimensions to 2
    np.random.seed(0)
    tsne = TSNE(n_components=2)
    train_data_features_reduced = tsne.fit_transform(X_SVD)

    zero_datapoints_array = [] #an array of the data points containing value 0
    one_datapoints_array = []# an array of the data points containing value 1

    # Iterate over the first 2000 samples of train_data_labels
    for i in range(0,2000):
        if train_data_labels[i] == 0:                   # extracting  0 is label for T-shirt
            zero_datapoints_array.append(train_data_features_reduced[i])
        elif train_data_labels[i] == 1:                   # extracting ones
            one_datapoints_array.append(train_data_features_reduced[i])



    zero_datapoints_array = np.array(zero_datapoints_array)
    one_datapoints_array = np.array(one_datapoints_array)

    def normalize(arr, max_val, n):
        a = np.divide(arr, max_val)
        return a + n

    zero_datapoints_normalized = normalize(zero_datapoints_array, 100, 1)
    one_datapoints_normalized = normalize(one_datapoints_array, 100, 1)


def decode(histogram):
    #encode & decodehere
    import qiskit
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

    # Define the quantum and classical registers
    q = QuantumRegister(16) # n is the number of qubits you want to use
    c = ClassicalRegister(16)

    # Create the quantum circuit
    qc = QuantumCircuit(q, c)

    # Iterate over each data point and apply rotations
    for i in range(len(train_data_features_reduced)):
        for j in range(2):
            qc.ry(train_data_features_reduced[i][j], q[j])

    # Measure the qubits and store the result in the classical register
    qc.measure(q, c)

    # Execute the circuit on a quantum backend
    backend = qiskit.Aer.get_backend('qasm_simulator')
    result = qiskit.execute(qc, backend, shots=1024).result()
    counts = result.get_counts()

    import matplotlib.pyplot as plt

    plt.bar(counts.keys(), counts.values(), color='g')
    plt.show()

def run_part1(image):
    #encode image into a circuit
    circuit=encode(image)

    #simulate circuit
    histogram=simulate(circuit)

    #reconstruct the image
    image_re=decode(histogram)

    return circuit,image_re

def run_part2(image):
    #part 2 cirq method 

    import cirq
    import numpy as np

    # Slice the arrays to only include the first num_samples of each class

    # Define the quantum circuit
    qnn = cirq.Circuit()
    qubits = cirq.LineQubit.range(8)

    # Iterate over each data point and apply rotations
    for i in range(len(one_datapoints_normalized)):
        for j in range(10):
            qnn.append(cirq.ry(one_datapoints_normalized[i][j]).on(qubits[j]))

    # Measure the qubits and store the result in a classical register
    qnn.append(cirq.measure(*qubits))

    # Execute the circuit on a quantum simulator
    simulator = cirq.Simulator()
    zero_result = simulator.run(qnn, repetitions=1024)

    # Define the quantum circuit
    qnn = cirq.Circuit()
    qubits = cirq.LineQubit.range(8)

    # Iterate over each data point and apply rotations
    for i in range(len(one_datapoints_normalized)):
        for j in range(2):
            qnn.append(cirq.ry(one_datapoints_normalized[i][j]).on(qubits[j]))

    # Measure the qubits and store the result in a classical register
    qnn.append(cirq.measure(*qubits))

    # Execute the circuit on a quantum simulator
    simulator = cirq.Simulator()
    one_result = simulator.run(qnn, repetitions=1024)

    qubits = cirq.LineQubit.range(16)
    circuit = cirq.Circuit()
    for i in range(len(zero_datapoints_normalized)):
        for j in range(2):
            circuit.append(cirq.ry(zero_datapoints_normalized[i][j]).on(qubits[j]))

    # Apply quantum gates
    circuit.append(cirq.CNOT(qubits[0], qubits[1]))
    circuit.append(cirq.CZ(qubits[0], qubits[2]))
    circuit.append(cirq.CNOT(qubits[1], qubits[3]))

    # Measure the qubits
    for i in range(16):
        circuit.append(cirq.measure(qubits[i], key=str(i)))


    #shape
    print (zero_datapoints_normalized.shape)
    print (train_data_labels.shape)

    # Now you can use the zero_data and one_data arrays in your Q
    from sklearn.svm import SVC
    clf = SVC()
    train_data_labels_reshaped = train_data_labels.reshape(-1, 1)
    print(len(train_data_labels_reshaped))

    num_samples = min(len(zero_datapoints_normalized), len(train_data_labels))
    zero_datapoints_normalized = zero_datapoints_normalized[:num_samples]
    train_data_labels = train_data_labels[:num_samples]

    threshold = 0.5
    import numpy as np
    # Convert continuous labels to 0s and 1s
    train_data_labels = np.where(train_data_labels > threshold, 1, 0)
    zero_datapoints_normalized = np.where(zero_datapoints_normalized > threshold, 1, 0)

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(zero_datapoints_normalized, train_data_labels, test_size=0.2, random_state=42)

    from sklearn.svm import OneClassSVM
    clf = OneClassSVM()
    clf.fit(X_train)

############################
#      END YOUR CODE       #
############################

test()
