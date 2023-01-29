import qiskit as qsk
import numpy as np
import skimage
from skimage.transform import resize

def make_circuit(image, n=16):
    circuit = qsk.QuantumCircuit(n,n)
    for i in range(n):
        for j in range(n):
            if i == j:
                circuit.rx(image[i][i],i)
            else:
                circuit.crx(image[i][j],i,j) #pixel (1,0)
    return(circuit)

def encode(image):
    image = resize(image, (16, 16))
    return(make_circuit(image))
    