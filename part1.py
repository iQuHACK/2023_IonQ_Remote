import qiskit
import numpy as np

NB_QUBITS = 10 # 2*n, no color info

def pixel_value_to_theta(pixel: int) -> float:
    return pixel / 255 * (np.pi/2)


def encode_qiskit(image):
    q = qiskit.QuantumRegister(NB_QUBITS)
    circuit = qiskit.QuantumCircuit(q)
    
    for i in range(NB_QUBITS):
        circuit.h(i)
        
    circuit.barrier()
    
    
    
    print(circuit)
    return circuit


def decode(histogram):
    if 1 in histogram.keys():
        image=[[0,0],[0,0]]
    else:
        image=[[1,1],[1,1]]
    return image


if __name__ == "__main__":
    pass