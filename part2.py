import numpy as np
import pickle
from collections import Counter
from sklearn.metrics import mean_squared_error
import qiskit
from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from part1 import encode, simulator

def histogram_to_category(histogram: dict) -> int:
    assert abs(sum(histogram.values())-1)<1e-8
    positive=0
    for key in histogram.keys():
        digits = bin(int(key))[2:].zfill(20)
        if digits[-1]=='0':
            positive+=histogram[key]
    return positive


def load_qasm(path: str) -> qiskit.QuantumCircuit:
    with open(path, 'r') as f:
        qasm=f.read()
    return qiskit.QuantumCircuit.from_qasm_str(qasm)


def split_train_test_data(images: np.ndarray, labels: np.ndarray, train_ratio: float=0.8) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nb_train = int(len(images)*train_ratio)
    train_images = images[:nb_train]
    train_labels = labels[:nb_train]
    test_images = images[nb_train:]
    test_labels = labels[nb_train:]
    return train_images, train_labels, test_images, test_labels


class QuantumCircuit:
    def __init__(self, circuit, backend=qiskit.Aer.get_backend("aer_simulator"), shots=1024):
        # This circuit will be parametrised by the weights of a upstream NN
        self.circuit = circuit
        self.theta = qiskit.circuit.Parameter('theta')
        self.backend = backend
        self.shots = shots
    
    def simulate(self, weights: np.ndarray):
        t_qc = qiskit.transpile(self.circuit, self.backend)
        qobj = qiskit.assemble(t_qc, shots=self.shots, parameter_binds = [{self.theta: weight} for weight in weights])
        job = self.backend.run(qobj)
        result = job.result().get_counts(self.circuit)
        
        counts = np.array(list(result.values()))
        states = np.array(list(result.keys())).astype(float)
                
        return result


class Functions:
    pass


class HybridNet:
    pass


class HybridClassifier:
    pass


def train_classifier():
    pass


def compute_loss():
    pass


def test_classifier(test_images: np.ndarray, test_labels: np.ndarray, classifier: qiskit.QuantumCircuit) -> Union[list, float]:
    nb_test = len(test_images)
    predictions = []
    for i in range(nb_test):
        image = test_images[i]
        _, prediction = run_part2(classifier, image)
        predictions.append(prediction)
    return predictions, mean_squared_error(test_labels, predictions)

def run_part2(pickle_path: str, image: np.ndarray) -> Union[qiskit.QuantumCircuit, int]:
    # Load the quantum classifier circuit
    with open(pickle_path, 'rb') as f:
        classifier=pickle.load(f)
    
    # Build circuit
    circuit = encode(image)
    circuit.append(classifier) ##
    
    # Simulate circuit
    histogram = simulator(circuit)
        
    # Convert histogram to category
    label = histogram_to_category(histogram)
    return circuit, label

if __name__ == "__main__":
    images = np.load('data/images.npy')
    labels = np.load('data/labels.npy')
    train_images, train_labels, test_images, test_labels = split_train_test_data(images, labels)

    classifier = load_qasm('part2.qasm') # To be integrated to a classical NN
    print(classifier)