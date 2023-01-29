import qiskit
from qiskit import quantum_info
from qiskit.execute_function import execute
from qiskit import BasicAer
import numpy as np
import pickle
import json
import os
from collections import Counter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import normalize
from typing import Dict, List
import matplotlib.pyplot as plt
import math
from qiskit import QuantumCircuit
from qiskit import QuantumCircuit, execute
from qiskit.tools.visualization import plot_histogram, circuit_drawer
from qiskit.providers.aer import UnitarySimulator
import numpy as np
from scipy.linalg import orth
import qiskit
from qiskit import quantum_info
from qiskit.execute_function import execute
from qiskit import BasicAer
import numpy as np
import pickle
import json
import os
from collections import Counter
from sklearn.metrics import mean_squared_error
from typing import Dict, List
import matplotlib.pyplot as plt
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
images=np.load('data/images.npy')
image = np.array(images[1])

def encode(image):
    fa = []
    for i in image:
        a = []
        for item in i:
            rv = int(item*1000)
            a.append(rv)
        fa.append(a)
    binary_image = np.array([[np.binary_repr(x) for x in row] for row in fa])
    final_list = []
    for i in binary_image:
        jl = []
        for j in i:
            a = int(j)
            jl.append(a)
        final_list.append(jl)
    # print(final_list)
    qc = QuantumCircuit(1)
    for i in final_list:
        for j in i:
            if j == 0:
                qc.x(0)
            if j == 1:
                qc.y(0)
            if j == 10:
                qc.z(0)
            if j == 11:
                qc.h(0)
    return qc

# encode(image)
def split_list(st, chunk_size):
    return [st[i:i + chunk_size] for i in range (0,len(st),chunk_size)]

def decode(histogram,circuit):
    data = histogram
    names = list(data.keys())
    values = list(data.values())
    plt.bar(range(len(data)), values, tick_label=names)
    plt.show()
    gates = []
    for gate in circuit.data:
        gates.append(gate[0].name)
    remade_list = []
    # for i in range(28):
    #     remade_list.append([])
    for i in gates:
        for j in i:
            if j == "x":
                # remade_list.append(0)
                remade_list.append(int('0',2)/1000)
            if j == "y":
                # remade_list.append(1)
                remade_list.append(int('1',2)/1000)
            if j == "z":
                # remade_list.append(10)
                remade_list.append(int('10',2)/1000)
            if j == "h":
                # remade_list.append(11)
                remade_list.append(int('11',2)/1000)
    whole_matrix = split_list(remade_list, len(remade_list) // 28)
    whole_matrix = np.array(whole_matrix)
    plt.imshow(whole_matrix)

def run_part1(image):
    #encode image into a circuit
    circuit=encode(image)
    #simulate circuit
    histogram=simulate(circuit)
    #reconstruct the image
    image_re=decode(histogram,circuit)
    return circuit,image_re
# qc.draw()
# gates = []
# for gate in qc.data:
#     gates.append(gate[0].name)
plt.imshow(image)
run_part1(image)

