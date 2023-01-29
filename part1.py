import cirq
import qiskit
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from qiskit import QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit import Aer, transpile

# using 16 components, so we have 4x4 for each image
pca = PCA(n_components=16)
MAX = 1


def _encoding(images):
    encoded_images = []
    for image in images:
        encoded_images.append(quantum_encode(image))
    return encoded_images

def flatten_images(images: list) -> list:
    """flatten each image to a row vector"""
    flattened_images = []
    for image in images:
        flattened_images.append(image.flatten())
    return np.array(flattened_images)

# preprocess when the script is invoked
def preprocessing():
    images = np.load("data/images.npy")
    # Normalize image data between 0 and 1
    global MAX
    MAX = np.max(images)
    images = images/MAX
    flat_images = flatten_images(images)
    # use PCA to achieve image reduction
    reduced_images = pca.fit_transform(flat_images)
preprocessing()

def _quantum_encode(image): 
    qc = QuantumCircuit(16,16)
    # Normalise data in image (might have to save normalisation to inverse later)
    scaler = MinMaxScaler()
    normalised = scaler.fit_transform(image.reshape(-1, 1))

    for i in range(len(normalised)):
        qc.ry(np.pi*float(normalised[i]), i)

    qc.measure(np.arange(0,16,1),np.arange(0,16,1))
    return qc


def encoder(image):
    image = image/MAX
    reduced_image = pca.transform(image.reshape(1, -1))
    input_circuit = _quantum_encode(reduced_image)
    return input_circuit

def decoder(histogram):
    shot_num = 10000
    amplist = np.zeros(16)
    keys = list(histogram.keys())
    for k in range(len(keys) - 1):
        binlist = [int(d) for d in str(keys[k])]
        amplist = amplist + [x * histogram[keys[k]]/shot_num for x in binlist]
    # invert the dimension reduction
    decoded_image = pca.inverse_transform(amplist.reshape(1, -1)).reshape(28, 28)
    print(amplist)
    return decoded_image

def run_part1(image):
    encoded_image = encoder(image)
    # run simulator and obtain histogram
    simulator = Aer.get_backend('aer_simulator')
    shot_num = 10000
    #index each qubit in the binary number
    num = 16
    result = simulator.run(encoded_image, shots=shot_num).result()
    counts = result.get_counts(encoded_image)
    decoded_image = decoder(counts)
    return encoded_image, decoded_image
     
def encode_cirq(image):
    circuit=cirq.Circuit()
    if image[0][0]==0:
        circuit.append(cirq.rx(np.pi).on(cirq.LineQubit(0)))
    return circuit

def encode_qiskit(image):
    q = qiskit.QuantumRegister(3)
    circuit = qiskit.QuantumCircuit(q)
    if image[0][0]==0:
        circuit.rx(np.pi,0)
    return circuit


def decode(histogram):
    if 1 in histogram.keys():
        image=[[0,0],[0,0]]
    else:
        image=[[1,1],[1,1]]
    return image
