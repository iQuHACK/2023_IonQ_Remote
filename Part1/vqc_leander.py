import numpy as np
from qiskit.algorithms.optimizers import COBYLA, ADAM, SPSA, SLSQP
from part1 import encoder
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.execute_function import execute
from qiskit import BasicAer
from collections import Counter
import qiskit
from typing import Dict, List
from multiprocessing import Process, Manager
from functools import partial
from sklearn.metrics import roc_auc_score


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

    qc = QuantumCircuit(16,16)
    qc = qc.compose(circuit)
    qc = qc.compose(vqc(n_qubits, n_layers, parameters))
    #simulate circuit
    histogram=simulate(qc)
    #convert histogram to category
    predict = histogram_to_category(histogram)
    return (label-predict)**2, predict

def cost_function(iterator, iterator_val, parameters):
    global BEST_LOSS
    global BEST_PARAMS
    global BEST_VALIDATION
    #print(parameters)
    def f(iterator):
        cost = []
        N = len(iterator)
        for i in range(N):
            image, label = iterator[i]
            l, predict = loss(image, label, parameters)
            cost.append([l, label, predict])
        return cost
    
    cost = parallelize('', f, iterator)
    res = np.mean(cost[:, 0])
    roc = roc_auc_score(cost[:, 1], cost[:, 2])

    cost_val = parallelize('', f, iterator_val)
    val = np.mean(cost_val[:, 0])
    roc_val = roc_auc_score(cost_val[:, 1], cost_val[:, 2])

    if val < BEST_VALIDATION:
        BEST_VALIDATION = val
        BEST_PARAMS = parameters
        np.save(open('params.npy', 'wb'), BEST_PARAMS)
    
    print(f'LOSS: {res} VAL: {val} ROC: {roc} ROCVAL: {roc_val}')
    return res  


def parallelize(process_name: str, f, iterator, *args):
    process = []
    iterator = list(iterator)
    N = len(iterator)

    def parallel_f(result, per, iterator, *args) -> None:
        '''
        Auxiliar function to help the parallelization

        Parameters:
            result : array_like
                It is a shared memory list where each result is stored.
            per : list[int]
                It is a shared memory list that contais the number of elements solved.
            iterator : array_like
                The function f is applied to elements in the iterator array.
        '''
        value = f(iterator, *args)              # The function f is applied to the iterator
        if value is not None:
            # The function may not return anything
            result += f(iterator, *args)        # Store the output into result array
        per[0] += len(iterator)                 # The counter is actualized
        print(per[0]/N, end='\r')
    
    result = Manager().list([])             # Shared Memory list to store the result
    per = Manager().list([0])               # Shared Memory to countability the progress
    f_ = partial(parallel_f,  result, per)  # Modified function used to create processes

    n = N//n_process                                                   # Number or processes
    for i_start in range(n_process):
        # Division of the iterator array into n smaller arrays
        j_end = n*(i_start+1) if i_start < n_process-1\
            else n*(i_start+1) + N % n_process
        i_start = i_start*n
        p = Process(target=f_, args=(iterator[i_start: j_end], *args)) 
        #print(f'Create Proces: {i_start}')     # Process creation
        p.start()                                                           # Initialize the process
        process.append(p)

    while len(process) > 0:
        p = process.pop(0)
        p.join()

    return np.array(result)


def store_intermediate_result(evaluation, parameter, cost, 
                              stepsize, accept):
    evaluations.append(evaluation)
    parameters.append(parameter)
    costs.append(cost)

parameters = []
costs = []
evaluations = []

BEST_PARAMS = None
BEST_LOSS = 1e3
BEST_VALIDATION = 1e3

n_process = 100

with open('../data/images.npy', 'rb') as f:
    images = np.load(f)
with open('../data/labels.npy', 'rb') as f:
    labels = np.load(f)*1

indexes = np.arange(len(images))
np.random.shuffle(indexes)

n_train = int(0.8 * len(images))
n_val = len(images) - n_train


images = images[indexes]
labels = labels[indexes]

iterator_train = list(zip(images[:n_train], labels[:n_train]))
iterator_val = list(zip(images[n_train:], labels[n_train:]))

optimizer = SPSA(maxiter=50)

p = np.random.random(3*n_qubits*n_layers)*2*np.pi

objective_function = lambda p: cost_function(iterator_train, iterator_val, p)
                                            
ret = optimizer.optimize(num_vars=3*n_qubits*n_layers, objective_function=objective_function, initial_point=p)

print("OPTIMIZATION COMPLETED! RESULT ---> {}".format(ret))


