import cirq
from qiskit import QuantumCircuit
import numpy as np


def encode_cirq(image):
    circuit=cirq.Circuit()
    if image[0][0]==0:
        circuit.append(cirq.rx(np.pi).on(cirq.LineQubit(0)))
    return circuit


def decode(hist):
    def bin_rep(x, n=8):
        t = "{0:b}".format(x)
        if len(t) < n:
            t = '0'*(n-len(t)) + t
        elif len(t) > n:
            t = t[len(t)-n:]
        return t
    dp = {}
    for i in range(32):
        for j in range(32):
            dp[bin_rep(i,5) + bin_rep(j,5)] = (i,j) 
    fdp = {}
    for k , v in dp.items():
        for i in range(64):
            fdp[bin_rep(i, 6) + k] = v
    img = np.zeros((28,28))
    for i in range(28):
        for j in range(28):
            t = [k for k , v in fdp.items() if v == (i,j)]
            p = -1
            res = '000000000000'
            for st in t:
                if st in hist:
                    if hist[st] > p:
                        p = hist[st]
                        res = st
            
            img[i][j] = int(res[:6],2)
    img = img*255/img.max()
    return img


def encode_qiskit(data):

    def bin_rep(x):
        t = "{0:b}".format(x)
        if len(t) < 8:
            t = '0'*(8-len(t)) + t
        return t
        data = data * 255 / data.max()
    data = data.astype(int)
    # M, N = data.shape
    # K = 2
    # L = 2
    # MK = M // K
    # NL = N // L
    # data = data[:MK*K, :NL*L].reshape(MK, K, NL, L).max(axis=(1, 3))
    data = data.flatten()

    qc = QuantumCircuit(16)
    # assume in 
    # I : 0 - 5
    # X 6 - 10
    # Y 11 - 15
    
    # Step 1 load blank image
    for i in range(6,16):
        qc.h(i)
    
    # Step 2:
    final_output = [] 
    n = 10
    q = 6
    num_qubits = n+q #8 qubits for pixels and 6 qubits for data 
    qc_image = QuantumCircuit(num_qubits) 

    # Create the pixel position qubits, and place them in superposition. 
    # qc_pos = QuantumCircuit(n) 
    for i in range(q, num_qubits):
        qc_image.h(i) 

    for idx in range(q): 
        qc_image.i(idx) 

    # Add the CNOT gates 
    for i , px in enumerate(data): 
        qc_image.x(num_qubits-1) 
        for idx, px_value in enumerate(bin_rep(px, q)): 
            if px_value=='1': 
                qc_image.ccx(num_qubits-1,num_qubits-2, idx) 
        qc_image.x(num_qubits-1) 
        qc_image.barrier() 

    return qc_image
    