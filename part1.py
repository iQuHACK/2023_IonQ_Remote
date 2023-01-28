import qiskit
import numpy as np

NB_QUBITS = 5


def pixel_value_to_theta(pixel: int) -> float:
    return pixel / 255 * (np.pi/2)


def encode_qiskit(image):
    q = qiskit.QuantumRegister(NB_QUBITS)
    circuit = qiskit.QuantumCircuit(q)
    
    for i in range(NB_QUBITS):
        pass
    
    
    if image[0][0]==0:
        circuit.rx(np.pi,0)
    return circuit


def decode(histogram):
    if 1 in histogram.keys():
        image=[[0,0],[0,0]]
    else:
        image=[[1,1],[1,1]]
    return image