# import Qiskit
from qiskit.extensions import Initialize
from qiskit import *
from qiskit import Aer, execute
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.library import RealAmplitudes, TwoLocal
from qiskit_ibm_runtime import QiskitRuntimeService, Options, Session, Sampler
from qiskit.providers.fake_provider import FakeManila
from qiskit_aer.noise import NoiseModel
import numpy as np
from sklearn import *
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC 
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram

def data_append(n1,n2, x1, x2):
    para_data = []
    for i in range(n1):
        for j in range(n2):
            if i<j:
                para_data.append(list(x1[i])+list(x2[j]))
    return para_data

img = np.load('images.npy')
lbl = np.load('labels.npy')
img3 = []
for ii in range(len(img)):
    img2 = img[ii].flatten()
    img3.append(img2)

def plot_matrix(A, title):
    """plots a given matrix."""
    # plot matrix
    plt.title(title)
    ax = plt.imshow(A, cmap='viridis')
    plt.colorbar(ax)
    plt.tight_layout()
    plt.show()

# split dataset
sample_train, sample_test, labels_train, labels_test = train_test_split(
     img3, lbl, test_size=0.7, random_state=22)

# reduce dimensions
n_dim = 14
pca = PCA(n_components=n_dim).fit(sample_train)
sample_train = pca.transform(sample_train)
sample_test = pca.transform(sample_test)

# standardize
std_scale = StandardScaler().fit(sample_train)
sample_train = std_scale.transform(sample_train)
sample_test = std_scale.transform(sample_test)

# Normalize
samples = np.append(sample_train, sample_test, axis=0)
minmax_scale = MinMaxScaler((-1, 1)).fit(samples)
sample_train = minmax_scale.transform(sample_train)
sample_test = minmax_scale.transform(sample_test)

# select 25 set of data for learning and 10 for test 
train_size = 25
sample_train = sample_train[:train_size]
labels_train = labels_train[:train_size]

test_size = 10
sample_test = sample_test[:test_size]
labels_test = labels_test[:test_size]


def encoder(image):
    circuit_1 = TwoLocal(7, 'ry', 'cx', reps=1, entanglement='pairwise', 
                           insert_barriers=True, parameter_prefix='x')
    qc = QuantumCircuit(7,7)
    qc += circuit_1.assign_parameters(image)
    return qc

def decoder(histogram):
    return None

def run_part1(img_data):
    circuit_1 = encoder(img_data)

    circuit_2 = RealAmplitudes(4, reps=1, entanglement='pairwise', 
                           insert_barriers=True, parameter_prefix='y')

    fidelity_circuit = circuit_1.copy()
    fidelity_circuit.append(circuit_2.inverse().decompose(), range(fidelity_circuit.num_qubits))
    fidelity_circuit.measure_all()
    k = data_append(len(img_data), 25, img_data, sample_train)

