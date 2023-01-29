import cirq
from qiskit import QuantumCircuit, Aer, assemble
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_quantum as tfq
import sympy
import seaborn as sns
import collections


# visualization tools
%matplotlib inline
import matplotlib.pyplot as plt
from cirq.contrib.svg import SVGCircuit


teamname="Quantum_QTCats"
task="part 1"

def encode_qiskit(image): #we make product amplitude encoding
    x_train = image
    #q = QuantumRegister(10) #2^10=1024 pixels and we need 28 x 28=784 pixels
    circuit=QuantumCircuit(10)
    values = np.ndarray.flatten(x_train) #array with the value of each pixel in the 28x28 image
    sum=np.sum(values**2)
    values=values/np.sqrt(sum)
    len_desire=2**10
    len_values=len(values)
    while len_values!=len_desire:
        values=np.append(values, 0) #we complete with zeros till reach the 1024 elements
        len_values+=1
    circuit.initialize(values)
    return circuit

def decode(histogram):
    image=np.zeros(784)
    for i in range(784):
        if i in histogram.keys():
            image[i]=histogram[i]
    image=image*255
    image = image.reshape(28, -1)
    return image

def run_part1(image):
    #encode image into a circuit
    image=np.array(image)
    circuit=encode_qiskit(image)

    #simulate circuit
    histogram=simulate(circuit)

    #reconstruct the image
    image_re=decode(histogram)
    plt.imshow(image_re)
    return circuit,image_re


