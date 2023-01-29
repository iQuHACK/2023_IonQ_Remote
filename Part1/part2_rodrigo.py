import numpy as np
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, qasm
import pickle
from qiskit.circuit import ParameterVector

class VQC():

    def __init__(self,n_layers,n_qubits,params):
        self.n_layers = n_layers
        self.n_qubits = n_qubits
        self.params = params

    def vqc(self):
        qr = QuantumRegister(self.n_qubits)
        qc = QuantumCircuit(qr)
        j = 0
        for l in range(self.n_layers):
            for i in range(self.n_qubits):
                qc.rx(self.params[j],i)
                j+=1
                qc.ry(self.params[j],i)
                j+=1
                qc.rz(self.params[j],i)
                j+=1
            for i in range(self.n_qubits-1):
                qc.cnot(i,i+1)
        return qc

n_layers = 1
n_qubits = 16
p = ParameterVector('$\theta$',3*n_layers*n_qubits)
my_vqc = VQC(n_layers,n_qubits,p)