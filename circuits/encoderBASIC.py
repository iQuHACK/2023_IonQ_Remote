import cirq
import numpy as np
import pickle
import json
import os

#submission to part 1, you should make this into a .py file
def encode(image):
    circuit=cirq.Circuit()
    epsilon = 0.1/255 #background is sometimes not exactly 0
    
    if len(image) < 28:
        rows_to_check = [0, 1]
        qubits = cirq.LineQubit.range(4)
    else:
        rows_to_check = [1, 7, 14, 21, 26] #rows we use to get info
        qubits = cirq.LineQubit.range(2*len(rows_to_check))
    
    #test qubit
    for row in rows_to_check:
        for i in range(len(image[0])):
            if image[row][i] >= epsilon:
                circuit.append(cirq.rx(np.pi/len(image[0]))(qubits[rows_to_check.index(row)])) #should represent % non-background values for a row in rows_to_check (first |rows_to_check| qubits)
                #check amount of "segments" (between 0 and 3) in each row in rows_to_check
                if i == 0:
                    circuit.append(cirq.rx(np.pi/3)(qubits[rows_to_check.index(row)+len(rows_to_check)])) #starts with non-background e.g. [[1,1],[1,1]]
                else:
                    if image[row][i-1] < epsilon:
                        circuit.append(cirq.rx(np.pi/3)(rows_to_check.index(row)+len(rows_to_check)))
    return circuit




def decode(histogram):
    if 1 in histogram.keys():
        image=[[0,0],[0,0]]
    else:
        image=[[1,1],[1,1]]
    return image



