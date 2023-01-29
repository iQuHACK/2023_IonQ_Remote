import numpy as np
from math import pi, log2, ceil

import qiskit as qk
import qiskit.circuit.library as crclib
from qiskit import QuantumCircuit
from qiskit import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import plot_histogram

def encode(image):
    image = np.array(image)*255
    qbitsx=ceil(log2(len(image)))
    qbitsy=ceil(log2(len(image[0])))
    qbits=qbitsx+qbitsy+1
    
    print(qbits)
    
    qc = QuantumCircuit(qbits,qbits)
    
    #prepare the state
    for i in range(qbits-1): 
        qc.h(i)
    qc.barrier()
    for x in range(len(image)):
        for y in range(len(image[x])):
            inst=crclib.RYGate(0.5*pi*image[x][y]).control(qbits-1)
            print(inst)
            qc.append(inst)
            #qc.barrier()
    
    qc=transpile(qc)
    
    return qc