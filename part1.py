from qiskit import IBMQ, QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit import execute, QuantumRegister
from qiskit.qasm import pi
from qiskit.tools.visualization import plot_histogram, circuit_drawer
from qiskit import execute, Aer, BasicAer
import numpy as np
import matplotlib.pyplot as plt
from qiskit.providers.aer import AerSimulator


def margolus(circ, t, c0, c1):
        circ.ry(np.pi/4,t)
        circ.cx(c0, t)
        circ.ry(np.pi/4,t)
        circ.cx(c1, t)
        circ.ry(-np.pi/4,t)
        circ.cx(c0, t)
        circ.ry(-np.pi/4,t)

def rccx(circ, t, c0, c1):
        circ.h(t)
        circ.t(t)
        circ.cx(c0, t)
        circ.tdg(t)
        circ.cx(c1, t)
        circ.t(t)
        circ.cx(c0, t)
        circ.tdg(t)
        circ.h(t)

def rcccx(circ, t, c0, c1, c2):
        circ.h(t)
        circ.t(t)
        circ.cx(c0, t)
        circ.tdg(t)
        circ.h(t)
        circ.cx(c1, t)
        circ.t(t)
        circ.cx(c2, t)
        circ.tdg(t)
        circ.cx(c1, t)
        circ.t(t)
        circ.cx(c2, t)
        circ.tdg(t)
        circ.h(t)
        circ.t(t)
        circ.cx(c0, t)
        circ.tdg(t)
        circ.h(t)


def ccry(circ, angle, t, c0, c1):
        circ.cu3(angle/2, 0, 0, c1, t)
        circ.cx(c1, c0)
        circ.cu3(-angle/2, 0, 0, c0, t)
        circ.cx(c1, c0)
        circ.cu3(angle/2, 0, 0, c0, t)

def mary(circ, angle, t, c0, c1):
        circ.ry(angle/4,t)
        circ.cx(c0, t)
        circ.ry(-angle/4,t)
        circ.cx(c1, t)
        circ.ry(angle/4,t)
        circ.cx(c0, t)
        circ.ry(-angle/4,t)
        circ.cx(c1, t)

def cccry(circ, angle, t, a, c0, c1, c2):
        margolus(circ, a, c1, c2)
        mary(circ, angle, t, a, c0)
        margolus(circ, a, c1, c2)

def mary_4(circ, angle, t, c0, c1, c2):
        circ.h(t)
        circ.t(t)
        circ.cx(c0,t)
        circ.tdg(t)
        circ.h(t)
        circ.cx(c1,t)
        circ.rz(angle/4,t)
        circ.cx(c2,t)
        circ.rz(-angle/4,t)
        circ.cx(c1,t)
        circ.rz(angle/4,t)
        circ.cx(c2,t)
        circ.rz(-angle/4,t)
        circ.h(t)
        circ.t(t)
        circ.cx(c0,t)
        circ.tdg(t)
        circ.h(t)

def mary_8(circ, angle, t, c0, c1, c2, c3, c4, c5, c6):
        circ.h(t)
        circ.t(t)
        rccx(circ, t, c0, c1)
        circ.tdg(t)
        circ.h(t)
        rccx(circ, t, c2, c3)
        circ.rz(angle/4,t)
        rcccx(circ, t, c4, c5, c6)
        circ.rz(-angle/4,t)
        rccx(circ, t, c2, c3)
        circ.rz(angle/4,t)
        rcccx(circ, t, c4, c5, c6)
        circ.rz(-angle/4,t)
        circ.h(t)
        circ.t(t)
        rccx(circ, t, c0, c1)
        circ.tdg(t)
        circ.h(t)

def c10ry(circ, angle, bin, target, anc, controls):

        clist = []

        for i in bin:
                clist.append(int(i))

        for i in range(len(clist)):
                if clist[i] == 0:
                        circ.x(controls[-i-1])

        margolus(circ, anc, controls[0], controls[1])
        circ.x(controls[0])
        circ.x(controls[1])
        margolus(circ, controls[1], controls[2], controls[3])
        circ.x(controls[2])
        circ.x(controls[3])
        margolus(circ, controls[3], controls[4], controls[5])
        circ.x(controls[4])
        circ.x(controls[5])
        
        margolus(circ, controls[5], controls[8], controls[9])
        margolus(circ, controls[4], controls[6], controls[7])
        margolus(circ, controls[2], controls[4], controls[5])
        margolus(circ, controls[0], controls[2], controls[3])

        mary_4(circ, angle, target, anc, controls[0], controls[1])

        margolus(circ, controls[0], controls[2], controls[3])
        margolus(circ, controls[2], controls[4], controls[5])
        margolus(circ, controls[4], controls[6], controls[7])
        margolus(circ, controls[5], controls[8], controls[9])
        
        circ.x(controls[5])
        circ.x(controls[4])
        margolus(circ, controls[3], controls[4], controls[5])
        circ.x(controls[3])
        circ.x(controls[2])
        margolus(circ, controls[1], controls[2], controls[3])
        circ.x(controls[1])
        circ.x(controls[0])
        margolus(circ, anc, controls[0], controls[1])

        for i in range(len(clist)):
                if clist[i] == 0:
                        circ.x(controls[-i-1])

def c10mary(circ, angle, bin, target, anc, controls):
        clist = []

        for i in bin:
                clist.append(int(i))

        for i in range(len(clist)):
                if clist[i] == 0:
                        circ.x(controls[-i-1])

        rccx(circ, anc, controls[4], controls[5])
        circ.x(controls[4])
        circ.x(controls[5])
        rccx(circ, controls[4], controls[6], controls[7])
        rccx(circ, controls[5], controls[8], controls[9])


        mary_8(circ, angle, target, anc, controls[0], controls[1], controls[2], controls[3], controls[4], controls[5])

        rccx(circ, controls[5], controls[8], controls[9])
        rccx(circ, controls[4], controls[6], controls[7])
        circ.x(controls[5])
        circ.x(controls[4])
        rccx(circ, anc, controls[4], controls[5])

        for i in range(len(clist)):
                if clist[i] == 0:
                        circ.x(controls[-i-1])

def encode_qiskit(image):
    """
    This code is for 10 pixel qubits + 1 color qubit + 1 ancilla
    Input:
        image:
            np.array
    Output:
        qc:
            quantum circuit initializing the frqi state
    """
    # 2-dimentional data convert to 1-dimentional array
    image = np.asarray(image).reshape(784)
    # change type
    image = image.astype('float64')
    # Normalization(0~pi/2)
    image /= image.max()
    image = np.arcsin(image)
    qubit = 12
    qc = QuantumCircuit(qubit)


    # apply hadamard gates
    qc.h(range(2,qubit))

    # apply c10Ry gates (representing color data)
    for i in range(len(image)):
            if image[i] != 0:
                    c10mary(qc, 2 * image[i], format(i, '010b'), 0, 1, [i for i in range(2,12)])


    qc.measure_all()
    return qc

def decode(counts):
    """
    This restores the image from given counts dictionary
    The code is for 10 pixel qubits + 1 color qubit
    """
    genimg = np.array([])
    for i in range(784):
            try:
                    genimg = np.append(genimg, [np.sqrt(counts[format(i, '010b')+'01']/sum(counts.values()))])
            except KeyError:
                    genimg = np.append(genimg, [0.0])
    # inverse nomalization
    genimg = 0.00392156862745098*genimg/genimg.max()
    # back to 2-dimentional data
    genimg = genimg.reshape((28,28))
    return genimg

def simulate(qc):
    aersim = AerSimulator()
    return(execute(qc, aersim, shots=64*784).result().get_counts())