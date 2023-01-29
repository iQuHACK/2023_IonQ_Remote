import qiskit as qk
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

from qiskit import QuantumCircuit, Aer, IBMQ
from qiskit import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import plot_histogram
from math import pi, ceil

            
def bitfield(n):
    return [1 if digit=='1' else 0 for digit in bin(n)[2:]]      

def angle(img):
    normal = np.max(img)
    theta = (np.pi/2) / normal * img
    return theta


# accepts square matrix
def encoder(img):
    N = len(img)
    thetas = angle(img)
    
    qunum = ceil(np.log2(N*N)) + 1
    qc = QuantumCircuit(qunum)

    for i in range(qunum):
        qc.h(i)

    qc.barrier()
    
    binary = prev_binary = 0
    
    for i in range(N):
        for j in range(N):
            change = bitfield(binary^prev_binary)[:qunum]
            for k, n in enumerate(change):
                n and qc.x(k)
                
            qc.mcry(thetas[i][j], [l for l in range(qunum - 1)], qunum - 1)
                       
            qc.barrier()
            # increment to next pixel
            prev_binary = binary
            binary += 1
            
    qc.measure_all()
    
    return qc

def simulate(qc):
    aer_sim = Aer.get_backend('aer_simulator')
    t_qc1 = transpile(qc, aer_sim)
    qobj = assemble(t_qc1, shots=8192*256)
    result = aer_sim.run(qobj).result()
    counts = result.get_counts(qc)
    
    return counts


def decoder(histogram):
    new_dict = defaultdict(lambda: np.zeros(2))

    for key, value in histogram.items():
        new_dict[key[1:]][int(key[0],2)] = value
    
    N = 32
    image = np.zeros([N, N])

    for key, val in new_dict.items():
        val/=sum(val)
        
        a, b = divmod(int(key,2), N)

        image[a][b] = np.arccos(np.sqrt(val[0]))
        
    return image

def run_part1(image):
    N_orig = len(image)

    img = np.zeros([32,32])

    for i in range(N_orig):
        for j in range(N_orig):
            img[i][j] = image[i][j]
    
    N = len(img)

    plt.imshow(img[0:N,0:N])
    plt.show()

    qc = encoder(img)
    
    histogram = simulate(qc)
    
    image = decoder(histogram)
    
    plt.imshow(image)
    plt.show()
    return image[:N_orig, :N_orig]
