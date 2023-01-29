import numpy as np
import pickle
from collections import Counter
from sklearn.metrics import mean_squared_error
import qiskit
from typing import Union

from part1 import encode, simulator

def histogram_to_category(histogram: dict) -> int:
    assert abs(sum(histogram.values())-1)<1e-8
    positive=0
    for key in histogram.keys():
        digits = bin(int(key))[2:].zfill(20)
        if digits[-1]=='0':
            positive+=histogram[key]
    return positive

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
    return circuit,label


def split_train_test_data(images: np.ndarray, labels: np.ndarray, train_ratio: float=0.8) -> Union[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nb_train = int(len(images)*train_ratio)
    train_images = images[:nb_train]
    train_labels = labels[:nb_train]
    test_images = images[nb_train:]
    test_labels = labels[nb_train:]
    return train_images, train_labels, test_images, test_labels


def train_classifier():
    pass


def test_classifier():
    pass


if __name__ == "__main__":
    images = np.load('data/images.npy')
    labels = np.load('data/labels.npy')
    train_images, train_labels, test_images, test_labels = split_train_test_data(images, labels)

    run_part2('part2.pickle', image)