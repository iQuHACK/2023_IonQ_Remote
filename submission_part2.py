# -*- coding: UTF-8 -*-

import qiskit
import submission_part1

# Provided by IonQ


def simulate(circuit: qiskit.QuantumCircuit) -> dict:
    """Simulate the circuit, give the state vector as the result."""
    backend = qiskit.BasicAer.get_backend('statevector_simulator')
    job = qiskit.execute_function.execute(circuit, backend)
    result = job.result()
    state_vector = result.get_statevector()

    histogram = dict()
    for i in range(len(state_vector)):
        population = abs(state_vector[i]) ** 2
        if population > 1e-9:
            histogram[i] = population

    return histogram


# Provided by IonQ
def histogram_to_category(histogram):
    """This function take a histogram representations of circuit execution results, and process into labels as described in 
    the problem description."""
    assert abs(sum(histogram.values())-1) < 1e-8
    positive = 0
    for key in histogram.keys():
        digits = bin(int(key))[2:].zfill(20)
        if digits[-1] == '0':
            positive += histogram[key]

    return positive


def run_part2(image):
    # load the quantum classifier circuit
    classifier = qiskit.QuantumCircuit.from_qasm_file(
        'submission_classifier.qasm')

    # encode image into circuit
    circuit = submission_part1.encode(image)

    # append with classifier circuit
    nq1 = circuit.width()
    nq2 = classifier.width()
    nq = max(nq1, nq2)
    qc = qiskit.QuantumCircuit(nq)
    qc.append(circuit.to_instruction(), list(range(nq1)))
    qc.append(classifier.to_instruction(), list(range(nq2)))

    # simulate circuit
    histogram = simulate(qc)

    # convert histogram to category
    label = histogram_to_category(histogram)

    if label > 0.5:
        label = False
    else:
        label = True

    return circuit, label
