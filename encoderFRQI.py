import numpy as np

import qiskit as qk
from qiskit import QuantumCircuit
from qiskit import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import plot_histogram
from math import pi,log2

qbits=9

def encode(image):
    image*=255
    qbitsx=ceil(log2(len(image)))
    qbitsy=ceil(log2(len(image[0])))
    qbits=qbitsx+qbitsy+1
    
    qc = QuantumCircuit(qbits)
    
    #prepare the state
    for i in qbits: 
        qc.h(i)
    qc.barrier()
    
    for x in range(len(image)):
        for y in range(len(image[x])):
            qc.cry(0.5*pi*image[x][y],np.arange(qbits-1),qbits-1)
            qc.barrier()
    
    qc.transpile()
    
    return qc