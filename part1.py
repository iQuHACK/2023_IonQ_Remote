import numpy as np
import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *
from qiskit  import Aer, QuantumCircuit

from qiskit.utils import QuantumInstance
from qiskit.opflow import AerPauliExpectation
from qiskit.circuit import Parameter
from qiskit.circuit.library import RealAmplitudes, ZZFeatureMap


import matplotlib.pyplot as plt
import os
import pandas as pd


from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import MinMaxScaler


def ansatz(num_qubits):
    return RealAmplitudes(num_qubits, reps=5)

# Maybe we dont need this
# def encode_cirq(image):
#     circuit=cirq.Circuit()
#     if image[0][0]==0:
#         circuit.append(cirq.rx(np.pi).on(cirq.LineQubit(0)))
#     return circuit

def encode_qiskit(image):
    # def flatten_images(images):
    #     flattened_images = []
    #     flattened_images.append(images.flatten())
    #     return np.array(flattened_images)
    # flat_images = flatten_images(np.array(image))
    encode = ZZFeatureMap(feature_dimension=len(image), entanglement='linear')

    # construct ansatz
    num_inputs = len(image)
    ansatz = RealAmplitudes(len(image), reps=1)

    params = encode.parameters

    # construct quantum circuit
    QC = QuantumCircuit(num_inputs)
    QC.append(encode, range(num_inputs))
    QC.append(ansatz, range(num_inputs))
    return QC, params


def decode(histogram):
    if 1 in histogram.keys():
        image=[[0,0],[0,0]]
    else:
        image=[[1,1],[1,1]]
    return image
