import cirq
import numpy as np
import pickle
import json
import os
import sys
from collections import Counter
from sklearn.metrics import mean_squared_error

if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    data_path = './data/'

#define utility functions

def simulate(circuit: cirq.Circuit) -> dict:
    """This function simulates a Cirq circuit (without measurement) and outputs results in the format of histogram.
    """
    simulator = cirq.Simulator()
    result = simulator.simulate(circuit)
    
    state_vector=result.final_state_vector
    
    histogram = dict()
    for i in range(len(state_vector)):
        population = abs(state_vector[i]) ** 2
        if population > 1e-9:
            histogram[i] = population
    
    return histogram


def histogram_to_category(histogram):
    """This function takes a histogram representation of circuit execution results, and processes into labels as described in
    the problem description."""
    # assert abs(sum(histogram.values())-1)<1e-8
    positive=0
    for key in histogram.keys():
        digits = bin(int(key))[2:].zfill(20)
        if digits[-1]=='0':
            positive+=histogram[key]
        
    return positive

def count_gates(circuit: cirq.Circuit):
    """Returns the number of 1-qubit gates, number of 2-qubit gates, number of 3-qubit gates...."""
    counter=Counter([len(op.qubits) for op in circuit.all_operations()])
    
    #feel free to comment out the following two lines. But make sure you don't have k-qubit gates in your circuit
    #for k>2
    for i in range(3,20):
        assert counter[i]==0
        
    return counter

def image_mse(image1,image2):
    # Using sklearns mean squared error:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    return mean_squared_error(255*image1,255*image2)

def test():
    #load the actual hackthon data (fashion-mnist)
    images=np.load(data_path+'/images.npy')
    labels=np.load(data_path+'/labels.npy')
    
    #test part 1

    n=len(images)
    mse=0
    gatecount=0

    for image in images:
        #encode image into circuit
        circuit,image_re=run_part1(image)

        #count the number of 2qubit gates used
        gatecount+=count_gates(circuit)[2]

        #calculate mse
        mse+=image_mse(image,image_re)

    #fidelity of reconstruction
    f=1-mse/n
    gatecount=gatecount/n

    #score for part1
    score_part1=f*(0.999**gatecount)
    
    #test part 2
    
    score=0
    gatecount=0
    n=len(images)

    for i in range(n):
        #run part 2
        circuit,label=run_part2(images[i])

        #count the gate used in the circuit for score calculation
        gatecount+=count_gates(circuit)[2]

        #check label
        if label==labels[i]:
            score+=1
    #score
    score=score/n
    gatecount=gatecount/n

    score_part2=score*(0.999**gatecount)
    
    print(score_part1, ",", score_part2, ",", data_path, sep="")

############################
#      YOUR CODE HERE      #
############################
from copy import copy
from itertools import product
import skimage
import matplotlib.pyplot as plt

images=np.load(data_path+'/images.npy')
labels=np.load(data_path+'/labels.npy')

# Wtih 10 position qubits we can exactly reproduce the images through the encoding
# and decoding procedures.  This requires 2**10 CX gates. The best tradeoff between
# accurately representing the state and minimizing the number of CX gates occures
# around 6 position qubits.
n_position_qubits = 6

##############
##  Part 1  ##
##############

def gray_code_derivative(nbits):
    sequence = []
    for k in range(1, nbits+1):
        sequence = sequence + [k] + sequence
    return sequence

def gray_code(nbits):
    derivative = gray_code_derivative(nbits)
    sequence = [[0]*nbits]
    for k in range(len(derivative)):
        newentry = copy(sequence[k])
        newentry[derivative[k]-1] ^= 1
        sequence.append(newentry)
    return sequence

def binary_code(nbits):
    return [list(a)[::-1] for a in product([0, 1], repeat=nbits)]

def generate_M_matrix(nbits):
    gray_code_sequence = gray_code(nbits)
    binary_code_sequence = binary_code(nbits)
    
    M = np.zeros(shape=(2**nbits, 2**nbits))
    for i in range(2**nbits):
        for j in range(2**nbits):
            M[i, j] = (-1)**sum([binary_code_sequence[i][k] * gray_code_sequence[j][k] for k in range(nbits)])
    return M

M = generate_M_matrix(n_position_qubits)

def compute_rotation_angles(image):
    global n_position_qubits
    global M
    
    pixel_intensities = image.flatten()
    pixel_intensities = np.array(list(pixel_intensities) + [0]*(2**n_position_qubits-len(pixel_intensities)))

    return np.matmul(np.transpose(M) / 2**n_position_qubits, pixel_intensities)

def encode(image):
    """Encode image via FRQI representation. Encoding is done with CX gates and
    single qubit rotation gates using the procedure from fig 2 of arXiv:quant-ph/0404089
    """
    global n_position_qubits
    if n_position_qubits < 10:
        downscaled_pixels = int(np.sqrt(2**n_position_qubits))
        image = skimage.transform.resize_local_mean(image, output_shape=(downscaled_pixels, downscaled_pixels))
    
    rescale_intensity_factor = np.max(image)
    image = skimage.exposure.rescale_intensity(image)
    image = image * np.pi / 2
    
    rotation_angles = compute_rotation_angles(image)
    gray_code_derivative_sequence = gray_code_derivative(n_position_qubits)
    
    image = image.flatten()
    circuit = cirq.Circuit()
    
    for i in range(1, n_position_qubits + 1):
        circuit.append(cirq.H.on(cirq.LineQubit(i)))
    
    for i in range(2**n_position_qubits - 1):
        circuit.append(cirq.ry(rads=rotation_angles[i]).on(cirq.LineQubit(0)))
        circuit.append(cirq.CX(
            cirq.LineQubit(n_position_qubits - gray_code_derivative_sequence[i]) + 1,
            cirq.LineQubit(0))
            )
    circuit.append(cirq.ry(rads=rotation_angles[len(rotation_angles) - 1]).on(cirq.LineQubit(0)))
    circuit.append(cirq.CX(cirq.LineQubit(1), cirq.LineQubit(0)))
    
    circuit.append(cirq.rx(rads=np.pi * rescale_intensity_factor).on(cirq.LineQubit(-1)))
    
    return circuit


def decode(histogram):
    global n_position_qubits
    
    if n_position_qubits == 10:
        N = 28
    else:
        N = int(np.sqrt(2**n_position_qubits))
    
    image = np.zeros((N, N))
    
    intensityindexcos = 0
    intensityindexsin = 0
    if 0 in histogram.keys():
        intensityindexcos += histogram[0]
    if 2**n_position_qubits in histogram.keys():
        intensityindexcos += histogram[2**n_position_qubits]
    if 2 * 2**n_position_qubits in histogram.keys():
        intensityindexsin += histogram[2 * 2**n_position_qubits]
    if 3 * 2**n_position_qubits in histogram.keys():
        intensityindexsin += histogram[3 * 2**n_position_qubits]
    
    intensity_scale = np.arccos(np.sqrt(intensityindexcos / (intensityindexcos + intensityindexsin))) / (np.pi/4)
    
    for i in range(N**2):
        row = i // N
        col = i % N

        if i in histogram.keys():
            lightprob = histogram[i]
            
            darkindex = i + 2**n_position_qubits
            if darkindex in histogram.keys():
                darkprob = histogram[i + 2**n_position_qubits]
            else:
                darkprob = 0
            
            total = lightprob + darkprob
            norm_lightprob = lightprob / total
            lightamp = np.sqrt(norm_lightprob)
            
            theta = np.arccos(lightamp)
            intensity = theta / (np.pi / 2)
            
            image[row, col] = intensity * intensity_scale
        else:
            image[row, col] = 0
    
    image = skimage.transform.resize_local_mean(image, output_shape=(28, 28))
    return image

def run_part1(image):
    #encode image into a circuit
    circuit=encode(image)

    #simulate circuit
    histogram=simulate(circuit)

    #reconstruct the image
    image_re=decode(histogram)

    return circuit,image_re


##############
##  part 2  ##
##############

N = min(28, int(np.sqrt(2**n_position_qubits)))
sample = skimage.transform.resize_local_mean(images[0], output_shape=(N,N))
weights = compute_rotation_angles(sample)[::-1]

def make_classifier(weights):
    global n_position_qubits
    
    gray_code_derivative_sequence = gray_code_derivative(n_position_qubits)
    
    circuit = cirq.Circuit()
    
    circuit.append(cirq.Rx(rads=-np.pi * weights[-1]).on(cirq.LineQubit(-1)))
    
    circuit.append(cirq.CX(cirq.LineQubit(1), cirq.LineQubit(0)))
    circuit.append(cirq.ry(rads=-weights[len(weights) - 1]).on(cirq.LineQubit(0)))
    for i in range(2**n_position_qubits - 1):
        circuit.append(cirq.ry(rads=-weights[i]).on(cirq.LineQubit(0)))
        circuit.append(cirq.CX(
            cirq.LineQubit(n_position_qubits - gray_code_derivative_sequence[i]) +1,
            cirq.LineQubit(0))
            )
    for i in range(1, n_position_qubits + 1):
        circuit.append(cirq.H.on(cirq.LineQubit(i)))
    
    for i in range(-1, n_position_qubits + 1):
        circuit.append(cirq.X.on(cirq.LineQubit(i)))
        circuit.append(cirq.measure(cirq.LineQubit(i), key='o{}'.format(i)))
    circuit.append(cirq.X.on(cirq.LineQubit(-2)).with_classical_controls(*['o{}'.format(k) for k in range(-1, n_position_qubits+1)]))
    return circuit

with open(os.getcwd()+'/quantum_classifier.pickle', 'wb') as f:
    pickle.dump(make_classifier(weights), f)

def simulate_part2(circuit: cirq.Circuit) -> dict:
    """This function simulates a Cirq circuit (without measurement) and outputs results in the format of histogram.
    """
    simulator = cirq.Simulator()
    result = simulator.run(circuit, repetitions=100)
    
    return result.data

def measurement_outcomes_to_category(measurement_outcomes):
    """This function takes a histogram representation of circuit execution results, and processes into labels as described in
    the problem description."""
    repetitions = measurement_outcomes.shape[0]
    return len(np.prod(np.array(measurement_outcomes), axis=1).nonzero()[0]) / repetitions


def run_part2(image):
    # load the quantum classifier circuit
    with open('quantum_classifier.pickle', 'rb') as f:
        classifier=pickle.load(f)
    
    #encode image into circuit
    circuit=encode(image)
    
    #append with classifier circuit
    
    circuit.append(classifier)
    
    #simulate circuit
    histogram=simulate(circuit)
    #measurement_outcomes = simulate_part2(circuit)
        
    #convert histogram to category
    #label=measurement_outcomes_to_category(measurement_outcomes)
    label = histogram_to_category(histogram)
    
    #thresholding the label, any way you want
    if label>0.5:
        label=1
    else:
        label=0
        
    return circuit,label

############################
#      END YOUR CODE       #
############################

test()
