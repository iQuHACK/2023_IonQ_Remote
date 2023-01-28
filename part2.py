# part 2

def encode(image):
    circuit=cirq.Circuit()
    if image[0][0]==0:
        circuit.append(cirq.rx(np.pi).on(cirq.LineQubit(0)))
    return circuit


def run_part2(image):

    #load the quantum classifier circuit
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
        
    return circuit
    return label
