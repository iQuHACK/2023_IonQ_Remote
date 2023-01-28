import qiskit as qk
from qiskit import QuantumCircuit, Aer, IBMQ
from qiskit import transpile, assemble
from qiskit.tools.jupyter import *
from qiskit.visualization import plot_histogram
from math import pi
import matplotlib.pyplot as plt 

theta = [0, pi / 8, pi / 4, 3 * pi / 8]
qc = QuantumCircuit(3)

qc.h(0)
qc.h(1)

qc.barrier()
# Pixel 1

qc.cry(theta[0], 0, 2)
qc.cx(0, 1)
qc.cry(-theta[0], 1, 2)
qc.cx(0, 1)
qc.cry(theta[0], 1, 2)

qc.barrier()
# Pixel 2

qc.x(1)
qc.cry(theta[1], 0, 2)
qc.cx(0, 1)
qc.cry(-theta[1], 1, 2)
qc.cx(0, 1)
qc.cry(theta[1], 1, 2)

qc.barrier()

qc.x(1)
qc.x(0)
qc.cry(theta[2], 0, 2)
qc.cx(0, 1)
qc.cry(-theta[2], 1, 2)
qc.cx(0, 1)
qc.cry(theta[2], 1, 2)


qc.barrier()

qc.x(1)

qc.cry(theta[3], 0, 2)
qc.cx(0, 1)
qc.cry(-theta[3], 1, 2)
qc.cx(0, 1)
qc.cry(theta[3], 1, 2)

qc.measure_all()

print(qc.draw())

aer_sim = Aer.get_backend('aer_simulator')
t_qc = transpile(qc, aer_sim)
qobj = assemble(t_qc, shots=4096)
result = aer_sim.run(qobj).result()
counts = result.get_counts(qc)
print(counts)
plot_histogram(counts)
plt.show()