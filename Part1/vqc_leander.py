import numpy as np
from qiskit.algorithms.optimizers import COBYLA, ADAM, SPSA
from part1 import encoder
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.execute_function import execute
from qiskit import BasicAer
from collections import Counter
import qiskit
from typing import Dict, List


def vqc(n_qubits, n_layers, params):
    qr = QuantumRegister(n_qubits)
    qc = QuantumCircuit(qr)
    j = 0
    for l in range(n_layers):
        for i in range(n_qubits):
            qc.rx(params[j],i)
            j+=1
            qc.ry(params[j],i)
            j+=1
            qc.rz(params[j],i)
            j+=1
        for i in range(n_qubits-1):
            qc.cnot(i,i+1)
    return qc




n_qubits = 16
n_layers = 1

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
    #assert abs(sum(histogram.values())-1)<1e-8
    positive=0
    for key in histogram.keys():
        digits = bin(int(key))[2:].zfill(20)
        if digits[-1]=='0':
            positive+=histogram[key]
        
    return positive

def loss(image, label, parameters):
    circuit=encoder(image)
    
    #append with classifier circuit
    qc = QuantumCircuit(16,16)
    qc = qc.compose(circuit)
    qc = qc.compose(vqc(n_qubits, n_layers, parameters))
    
    #simulate circuit
    histogram=simulate(qc)
        
    #convert histogram to category
    predict = histogram_to_category(histogram)
    #print("Prediction: ", predict)
    #print("Label: ", label)
    return (label-predict)**2

def cost_function(images, labels, parameters):
    cost = []
    labels = labels*1
    N = len(images)
    for i in range(N):
        print(f'{i}/{N}', end='\r')
        cost.append(loss(images[i], labels[i], parameters))
    return np.mean(cost)


def store_intermediate_result(evaluation, parameter, cost, 
                              stepsize, accept):
    evaluations.append(evaluation)
    parameters.append(parameter)
    costs.append(cost)

parameters = []
costs = []
evaluations = []

with open('../data/images.npy', 'rb') as f:
    images = np.load(f)
with open('../data/labels.npy', 'rb') as f:
    labels = np.load(f)

optimizer = SPSA(maxiter=50,callback=store_intermediate_result)

p = np.random.random(3*n_qubits*n_layers)

objective_function = lambda p: cost_function(images[0:50],labels[0:50],p)
                                            
ret = optimizer.optimize(num_vars=3*n_qubits*n_layers, objective_function=objective_function, initial_point=p)

print("OPTIMIZATION COMPLETED! RESULT ---> {}".format(ret))