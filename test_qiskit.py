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
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit import Aer, QuantumCircuit
from qiskit.circuit.library import TwoLocal
from IPython.display import display
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pickle

shots = 50000
# Create the quantum feature map (in this case a TwoLocal)
qc = TwoLocal(7, ['ry', 'rx'], 'cx', reps=1, entanglement='linear', 
                           insert_barriers=True, parameter_prefix='x')

def encoder(image):
    encd = QuantumCircuit(7,7)
    encd += qc.assign_parameters(image*np.pi)
    encd.measure(range(7), range(7))
    display(encd.decompose().draw(output = "mpl", fold = -1))

    backend = Aer.get_backend('aer_simulator')
    counts = backend.run(encd.decompose(), shots = shots).result().get_counts()
    display(plot_histogram(counts))
    return encd, counts

def decoder(histogram):
    recImg = []
    for ii in range(16):
        u = []
        for jj in range(8):
            if str(format(8*ii+jj,'07b')) in histogram:
                u.append(histogram[format(8*ii+jj,'07b')]/shots)
            else:
                u.append(0)   
        recImg.append(u)
    plt.imshow(recImg)

def run_part1(image):
    circ, hist = encoder(image)
    decoder(hist)
    

def run_part2(image):
    
    if type(image[0]) == "array" and len(image[0]) != 28:
        infile = open("part2.pickle",'rb')
        qsvc = pickle.load(infile)
        infile.close()
        # reduce dimensions
        n_dim = 28
        pca = PCA(n_components=n_dim).fit(image)
        image = pca.transform(image)

        # standardize
        std_scale = StandardScaler().fit(image)
        image = std_scale.transform(image)

        # Normalize
        minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
        image = minmax_scale.transform(image)
    
    encd = encoder(image)
    return qsvc.predict(image)


############################
#      END YOUR CODE       #
############################

test()
