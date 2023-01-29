def encode(image):
    q = qiskit.QuantumRegister(3)
    qc = qiskit.QuantumCircuit(q)
    img = images
    image = np.array(images)
    if image[0][0]==0:
        circuit.rx(np.pi,0)
    else:
        theta = np.pi #  according to line circuit.rx(np.pi,0)
        qc.h(0)
        qc.h(1)

        qc.barrier()
    

        qc.cry(theta,0,2)
        qc.cx(0,1)
        qc.cry(-theta,1,2)
        qc.cx(0,1)
        qc.cry(theta,1,2)

        qc.barrier()


        qc.x(1)
        qc.cry(theta,0,2)
        qc.cx(0,1)
        qc.cry(-theta,1,2)
        qc.cx(0,1)
        qc.cry(theta,1,2)
  
        qc.barrier()

        qc.x(1)
        qc.x(0)
        qc.cry(theta,0,2)
        qc.cx(0,1)
        qc.cry(-theta,1,2)
        qc.cx(0,1)
        qc.cry(theta,1,2)


        qc.barrier()

        qc.x(1)
   
        qc.cry(theta,0,2)
        qc.cx(0,1)
        qc.cry(-theta,1,2)
        qc.cx(0,1)
        qc.cry(theta,1,2)

        qc.measure_all()

        qc.draw()
    return qc
def decode(histogram):
    if 1 in histogram.keys():
        image=np.array([[0,0],[0,0]])
    else:
        image=np.array([[1,1],[1,1]])
    return image


def run_part2(image):

    #loade the quantum classifier circuit
    classifier=qiskit.QuantumCircuit.from_qasm_file('quantum_classifier.qasm')
    
    #encode image into circuit
    qc=encode(image)
    
    #append with classifier circuit
    nq1 = qc.width()
    nq2 = classifier.width()
    nq = max(nq1, nq2)
    qc = qiskit.QuantumCircuit(nq)
    qc.append(qc.to_instruction(), list(range(nq1)))
    qc.append(classifier.to_instruction(), list(range(nq2)))
    
    #simulate circuit
    histogram=simulate(qc)
        
    #convert histogram to category
    label=histogram_to_category(histogram)
        
    return qc,label
#score