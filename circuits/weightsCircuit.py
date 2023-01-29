import numpy as np

import qiskit as qk
from qiskit import QuantumCircuit
from qiskit import transpile, assemble
from math import pi, log2, ceil

#qbits=9


def encode(w): 
    qubits = len(w)
    qc = QuantumCircuit(qubits)
    
    for i in range(len(w)):
        qc.ry(pi*w[i],qubits[i])
    
    qc=transpile(qc)
    return qc