"""
The following code is a quantum image compression and decompression code that utilizes qiskit library.
It consists of several functions:

- simulate: takes a QuantumCircuit and returns a histogram of the statevector simulation result
- basis_states_probs: takes a histogram and returns the probabilities of all basis states
- reduze_size: reduces the size of the image
- encoder: encoding the image into a QuantumCircuit
- apply_decoder: applies the decoder on the reduced image
- decoder: decoding the histogram and return the image
"""
import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, Aer, transpile
from sklearn.metrics import mean_squared_error
from scipy.signal import convolve2d
from qiskit import quantum_info
from qiskit.execute_function import execute
from qiskit import BasicAer
import qiskit


def simulate(circuit: qiskit.QuantumCircuit) -> dict:
    """Simulate the circuit, give the state vector as the result."""
    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(circuit, backend)
    result = job.result()
    state_vector = result.get_statevector()
    
    histogram = dict()
    for i in range(len(state_vector)):
        population = abs(state_vector[i]) ** 2
        if population > 1e-9:
            histogram[i] = population
    
    return histogram

def basis_states_probs(counts):
    n = N_qubits
    return np.array([counts[vals] if counts.get(vals) is not None else 0 for vals in range(2**n)])

N_qubits = 16
values = np.array([ 
        0.27368858,  0.58202206,  0.12278033,  0.16740516,  0.38300442,
        0.17311421,  0.74180642,  0.77530172,  0.36415411,  1.09784771,
        0.79475861,  0.8381416 ,  0.89169281,  0.60547951, -0.13627254,
        0.64126044,  0.03553788,  0.97011833,  0.56889429,  0.21881036,
        0.41086148,  0.8060775 ,  0.55583887,  0.35244643,  0.75814781,
        0.40201306,  0.32262632,  0.49018184,  0.21012353,  0.31632245,
        0.48162911,  0.51701593,  1.01339176,  0.65412448,  0.46153903,
        0.70612024,  0.85376593,  0.70586454,  0.93835183,  0.7420828 ,
        0.92714791,  0.92789248,  0.10592274,  0.31667188,  0.59649797,
        0.10349468,  0.15288158,  0.59900233,  0.04632818,  0.48428953,
        0.19115525,  0.44538069,  0.38163836,  0.45138543,  0.65266545,
        0.83181461,  0.48835976,  0.34839669,  0.52279195,  0.49362643,
        0.62204067,  0.04987149,  0.69562555,  0.59832698,  0.10525147,
        0.38684568,  0.05195033,  0.12704839,  0.16099481,  0.86897043,
        0.43007063,  0.3139563 ,  0.71997524, -0.06254781,  0.84168987,
        0.27159143,  0.89167218,  0.62909926,  0.99054447,  0.74240935,
        0.0804703 ,  0.96066017,  0.88827054, -0.03132508,  0.54691696,
        0.61928724,  0.81545823,  0.41028532, -0.01744999,  0.63139688,
        0.96904866,  0.92977478,  0.40221932,  0.04401873,  0.57545908,
        0.01848986,  0.17006147,  0.54544609,  0.1164381 ,  0.07652701,
        0.62252829,  0.53195467,  0.33766771,  0.41798003,  0.11771296,
        0.62046955,  0.95366884,  1.13772555,  0.17542721,  0.96133482,
        0.4404973 ,  0.96692143, -0.00156096,  0.79830233,  0.20825297,
        0.08530044,  0.22016309,  0.01050147,  0.79674154,  0.72825014,
        0.43271133,  0.08274834,  0.3796315 ,  0.68780477,  0.44599441,
        0.76751076,  0.32092582,  0.65363651,  0.19371235,  0.71898758,
        0.13310255,  0.23751939,  0.34004462,  0.08432172,  0.24408651,
        0.15584135,  0.88246919,  0.73551328,  0.4075967 ,  0.89325029,
        0.34628271,  0.46622663, -0.19860429,  0.70766031,  0.10818171,
        0.13720818,  0.9007084 
    ])

kernel_encoder = np.reshape(values[:7*7], (7, 7))
kernels_decoder = np.reshape(values[7*7:], (2, 7, 7))

max_factor = 0.00392157


def reduze_size(image):
    """Reduces the size of the image from 24x24 to 4x4."""
    Nx, Ny = image.shape
    kx, ky = kernel_encoder.shape
    Dx = Nx//kx
    Dy = Ny//ky
    new_image = np.zeros((Dx, Dy))

    for x in range(Dx):
        for y in range(Dy):
            new_image[x, y] = np.sum(kernel_encoder * image[kx*x: kx*(x +1), ky*y: ky*(y+1)])
            
    new_image = new_image/np.max(new_image)*np.max(image)
    return new_image


def encoder(image):
    """Encoding the image into a QuantumCircuit."""

    image = image/np.max(image) if np.max(image) > 0 else image
    image_reduzed = reduze_size(image)
    q_register = QuantumRegister(N_qubits)
    c_register = ClassicalRegister(N_qubits)
    qc = QuantumCircuit(q_register, c_register)
    
    data = np.ndarray.flatten(image_reduzed) * np.pi
    for i, d in enumerate(data):
        qc.ry(d, q_register[i])
        qc.measure(q_register[i], c_register[i])
    
    return qc

def apply_decoder(kernel, image_resized):
    """Applies the decoder to the image."""
    Nx, Ny = image_resized.shape
    kx, ky = kernel.shape
    Dx = Nx*kx
    Dy = Ny*ky
    new_image = np.zeros((Dx, Dy))

    for x in range(Nx):
        for y in range(Ny):
            new_image[kx*x: kx*(x +1), ky*y: ky*(y+1)] = kernel * image_resized[x, y]
    return new_image

def decoder(histogram):
    """Decodes the histogram into an image."""
    probs = basis_states_probs(histogram)
    q_probs = []

    for q in range(1, N_qubits+1):
        q_probs.append(np.sum(np.array(np.split(probs,2**q))[np.arange(0,2**q,2)]))

    q_probs = np.array(q_probs)[::-1]
    data = np.arccos(np.sqrt(q_probs))*2/np.pi
    image_decoded = np.resize(data, (4, 4))
    result = image_decoded.copy()
    result = apply_decoder(kernels_decoder[0], result)

    def layer(kernel, M):
        res = convolve2d(kernel, np.arctan(M))
        nx, ny = res.shape
        res = res[nx//2-14:nx//2+14, ny//2-14:ny//2+14]
        return res 

    for kernel in kernels_decoder[1:]:
        result += layer(kernel, result)
    result /= np.max(result) * np.max(image_decoded)
    nx, ny = result.shape
    result = result[nx//2-14:nx//2+14, ny//2-14:ny//2+14]
    return result * max_factor
    

def run_part1(image):
    circuit = encoder(image)
    histogram = simulate(circuit)
    image = decoder(histogram)
    return circuit, image