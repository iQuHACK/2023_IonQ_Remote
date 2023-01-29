import pickle
import part1 as pt1
import cirq
import qiskit
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit import Aer, transpile
from qiskit import BasicAer, execute

with open('part2.pickle') as f:
    classifier = pickle.load(f)


def label_probability(results):
    """Converts a dict of bitstrings and their counts,
    to parities and their counts"""
    shots = sum(results.values())
    probabilities = {0: 0, 1: 0}
    for bitstring, counts in results.items():
        hamming_weight = sum(int(k) for k in list(bitstring))
        label = (hamming_weight+1) % 2
        probabilities[label] += counts / shots
    #print('probabilities calculated')
    return probabilities

def run_part2(image):
    input_circuit = pt1.encoder(image)
    full_circuit = input_circuit.compose(classifier)
    full_circuit.measure_all() #Full circuit will have variation set from pickle
    
    backend = BasicAer.get_backend('qasm_simulator')
    results = execute(full_circuit, backend).result()
    label = label_probability(results.get_counts())
    
    return label, full_circuit
