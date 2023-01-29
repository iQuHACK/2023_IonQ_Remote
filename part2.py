import numpy as np
import pickle
import json
import os
from collections import Counter
from sklearn.metrics import mean_squared_error

from part1 import encode

def histogram_to_category(histogram):
    assert abs(sum(histogram.values())-1)<1e-8
    positive=0
    for key in histogram.keys():
        digits = bin(int(key))[2:].zfill(20)
        if digits[-1]=='0':
            positive+=histogram[key]
        
    return positive

def run_part2(image):

    #loade the quantum classifier circuit
    with open('part2.pickle', 'rb') as f:
        classifier=pickle.load(f)
    
    #encode image into circuit
    circuit=encode(image)
    
    #append with classifier circuit
    
    circuit.append(classifier)
    
    #simulate circuit
    histogram=simulate(circuit)
        
    #convert histogram to category
    label=histogram_to_category(histogram)
        
    return circuit,label

if __name__ == "__main__":
    pass