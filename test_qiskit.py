# team name IonCoffee
# oana.bazavan@physics.ox.ac.uk
# challenge ionq remote
import qiskit
from qiskit import quantum_info, QuantumCircuit
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
data_path = "data"


def simulate(circuit):
    backend = backend = BasicAer.get_backend('qasm_simulator')
    job = execute(circuit, backend, shots = 10000)
    return job.result().get_counts()


def histogram_to_category(counts):
    # predicts 0 (not a tshirt) or 1 (t-shirt) from histogram of circuit measurements
    # majority vote
    zeros = 0
    ones = 0
    for count in counts:
        for bit in count:
            if bit == '0':
                zeros += counts[count]
            if bit == '1':
                ones += counts[count]
    if zeros > ones:
        prediction = 0
    else:
        prediction = 1
    return prediction

def count_gates(circuit: qiskit.QuantumCircuit) -> Dict[int, int]:
    """Returns the number of gate operations with each number of qubits."""
    counter = Counter([len(gate[1]) for gate in circuit.data])
    #feel free to comment out the following two lines. But make sure you don't have k-qubit gates in your circuit
    #for k>2
    #for i in range(2,20):
    #    assert counter[i]==0
        
    return counter


def image_mse(image1,image2):
    # Using sklearns mean squared error:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    return mean_squared_error(image1,image2)

def test():
    #load the actual hackthon data (fashion-mnist)
    images=np.load('data/images.npy')
    labels=np.load('data/labels.npy')
    
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
        mse+=image_mse(image,image_re) # we didnt normalise the image as per the original problem

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
    n_components = 8
    num_qub = 4
    principal_components = np.load('data/principal_components.npy')
    xx = image.flatten()
    x_principal_comps = np.zeros([n_components])
    for i in range(n_components):
        x_principal_comps[i] = np.dot(xx, principal_components[i])
    
    desired_state = np.zeros(2**num_qub)
    for i,comp in enumerate(x_principal_comps):
        if comp>0:
              desired_state[i]= comp
        else:
            desired_state[i+8] = comp
 
    desired_state = np.array(desired_state)/np.linalg.norm(desired_state)

    qc_LI = QuantumCircuit(num_qub, num_qub)
    qc_LI.initialize(desired_state, [0,1,2,3])
    qc_LI_width = qc_LI.width()
    return qc_LI.decompose(reps=10)

def decode(histogram):
    from collections import OrderedDict
    od = OrderedDict(sorted(histogram.items()))
    Counts = np.zeros(2**num_qub)
    total_Counts = 0.
    
    for key in od.keys():
        Counts[int(key[0:4],2)] = od[key]
        total_Counts +=od[key]
    
    amplitudes = np.sqrt(Counts/total_Counts)
    reconst_principle_vals = np.zeros(8)

    for i in range(8):
        if (amplitudes[i]>amplitudes[i+8]):
            reconst_principle_vals[i] = amplitudes[i]
        else:
            reconst_principle_vals[i] = -1*amplitudes[i+8] 
    image_re = np.zeros(28*28)    
    
    for i,a in enumerate(reconst_principle_vals):
        image_re +=  a*principal_components[i,:]
        
    image_re = image_re.reshape(28,28)
    return image_re

def run_part1(image):
    global num_qub 
    num_qub = 4 #3 for amplitudes 1 for sign
    global n_components
    n_components = 8
    global principal_components
    principal_components = np.load("data/principal_components.npy")
    circuit = encode(image)
    #circuit.measure_all()
    histogram = simulate(circuit)
    image_re = decode(histogram)
    return circuit, image_re

def run_part2(image):
    # load the quantum classifier circuit
    classifier=qiskit.QuantumCircuit.from_qasm_file('quantum_classifier.qasm')
    #classifier.measure_all()
    #encode image into circuit
    circuit=encode(image)
    circuit.draw()
    #append with classifier circuit
    nq1 = circuit.width()
    #print(nq1)
    nq2 = classifier.width()
    #print(nq2)
    nq = max(nq1, nq2)
    qc = qiskit.QuantumCircuit(4)
    qc.compose(circuit, inplace = True)
    qc.compose(classifier, inplace = True)
    #qc.append(circuit.to_instruction(), list(range(4)), list(range(4)))
    #qc.append(classifier.to_instruction(), list(range(4)), list(range(4)))
    
    #simulate circuit
    histogram=simulate(qc)
    #convert histogram to category
    label=histogram_to_category(histogram)
        
    return circuit,label

############################
#      END YOUR CODE       #
############################

test()
