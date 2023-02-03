
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit import Aer, QuantumCircuit
from qiskit.circuit.library import TwoLocal
from IPython.display import display
import numpy as np

shots = 50000
# Create the quantum feature map (in this case a TwoLocal)
qc = TwoLocal(7, ['ry', 'rx'], 'cx', reps=1, entanglement='linear', 
                           insert_barriers=True, parameter_prefix='x')

def encoder(image):
    encd = QuantumCircuit(7,7)
    encd += qc.assign_parameters(image*np.pi)
    encd.measure(range(7), range(7))
    display(encd.decompose().draw(output = "mpl", fold = -1))

    backend = Aer.get_backend('aer_simulator')
    counts = backend.run(encd.decompose(), shots = shots).result().get_counts()
    display(plot_histogram(counts))
    return encd, counts

def decoder(histogram):
    recImg = []
    for ii in range(16):
        u = []
        for jj in range(8):
            if str(format(8*ii+jj,'07b')) in histogram:
                u.append(histogram[format(8*ii+jj,'07b')]/shots)
            else:
                u.append(0)   
        recImg.append(u)
    plt.imshow(recImg)

def run_part1(image):
    circ, hist = encoder(image)
    decoder(hist)