import cirq
import numpy as np
import pickle
import json
import os
from collections import Counter
from sklearn.metrics import mean_squared_error

import qiskit
from qiskit import quantum_info
from qiskit.execute_function import execute
from qiskit import BasicAer
from typing import Dict, List, Union
import matplotlib.pyplot as plt

#define utility functions

def simulate(circuit: Union[cirq.Circuit, qiskit.QuantumCircuit], backend_='qiskit') -> dict:
    """This funcion simulate a cirq circuit (without measurement) and output results in the format of histogram.
    """
    if backend_ == 'circ':
        simulator = cirq.Simulator()
        result = simulator.simulate(circuit)

        state_vector=result.final_state_vector

        histogram = dict()
        for i in range(len(state_vector)):
            population = abs(state_vector[i]) ** 2
            if population > 1e-9:
                histogram[i] = population

        return histogram
    
    elif backend_ == 'qiskit':
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

    else:
        raise Exception()


def histogram_to_category(histogram):
    """This function take a histogram representations of circuit execution results, and process into labels as described in 
    the problem description."""
    assert abs(sum(histogram.values())-1)<1e-8
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
    for i in range(2,20):
        assert counter[i]==0
        
    return counter

def image_mse(image1,image2):
    # Using sklearns mean squared error:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    return mean_squared_error(image1, image2)