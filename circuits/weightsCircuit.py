import numpy as np

import qiskit as qk
from qiskit import QuantumCircuit
from qiskit import transpile, assemble
from math import pi, log2, ceil

qbits=9

def encode(w):
    # TODO should encode the weights matrix which is just 16 parameters
    
    
    qc = QuantumCircuit(qbits)
    
    #prepare the state
    # TODO: iterate
    
    qc=transpile(qc)
    
    return qc