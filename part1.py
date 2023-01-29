def preprocess(images):
    images = np.round(images*(2**16)/8)*8
    images /= 255
    images = (np.pi/2)*images
    return images
new_images = preprocess(images)

#submission to part 1, you should make this into a .py file
n=len(new_images)
mse=0
gatecount=0

# Functions 'encode' and 'decode' are dummy.
def encode(image):
    
    #Implementation of MRQI fro 28x28 images
    
    qubit = 12
    q = qiskit.QuantumRegister(qubit)
    qc = qiskit.QuantumCircuit(q)
    
    #Necessary gates for the whole circuit
    
    def rccx(circ, t, c0, c1):
        circ.h(t)
        circ.t(t)
        circ.cx(c0, t)
        circ.tdg(t)
        circ.cx(c1, t)
        circ.t(t)
        circ.cx(c0, t)
        circ.tdg(t)
        circ.h(t)

    def rcccx(circ, t, c0, c1, c2):
        circ.h(t)
        circ.t(t)
        circ.cx(c0, t)
        circ.tdg(t)
        circ.h(t)
        circ.cx(c1, t)
        circ.t(t)
        circ.cx(c2, t)
        circ.tdg(t)
        circ.cx(c1, t)
        circ.t(t)
        circ.cx(c2, t)
        circ.tdg(t)
        circ.h(t)
        circ.t(t)
        circ.cx(c0, t)
        circ.tdg(t)
        circ.h(t)
        
    def mary_8(circ, angle, t, c0, c1, c2, c3, c4, c5, c6):
        angle = float(angle)
        #print(angle)
        circ.h(t)
        circ.t(t)
        rccx(circ, t, c0, c1)
        circ.tdg(t)
        circ.h(t)
        rccx(circ, t, c2, c3)
        circ.rz(angle/2,t)
        rcccx(circ, t, c4, c5, c6)
        circ.rz(-angle/2,t)
        rccx(circ, t, c2, c3)
        circ.rz(angle/2,t)
        rcccx(circ, t, c4, c5, c6)
        circ.rz(-angle/2,t)
        circ.h(t)
        circ.t(t)
        rccx(circ, t, c0, c1)
        circ.tdg(t)
        circ.h(t)
        
    def c10mary(circ, angle, bin, target, anc, controls):
        clist = []

        for i in bin:
            clist.append(int(i))

        for i in range(len(clist)):
            if clist[i] == 0:
                circ.x(controls[-i-1])

        rccx(circ, anc, controls[4], controls[5])
        circ.x(controls[4])
        circ.x(controls[5])
        rccx(circ, controls[4], controls[6], controls[7])
        rccx(circ, controls[5], controls[8], controls[9])


        mary_8(circ, angle, target, anc, controls[0], controls[1], controls[2], controls[3], controls[4], controls[5])

        rccx(circ, controls[5], controls[8], controls[9])
        rccx(circ, controls[4], controls[6], controls[7])
        circ.x(controls[5])
        circ.x(controls[4])
        rccx(circ, anc, controls[4], controls[5])

        for i in range(len(clist)):
                if clist[i] == 0: #https://arxiv.org/pdf/2110.15672.pdf
                        circ.x(controls[-i-1])
                        
    
    #Apply the whole ciruit with the reshpaed image
    
    image = image.flatten().astype('float64')
                        
    qc.h(range(2,qubit))
    
    
    for i in range(len(image)): #NOT SURE ABOUT THIS
        if image[i] != 0:
            #print(t)
            c10mary(qc, 2 * image[i], format(i, '010b'), 10, 11, [i for i in range(0,10)])
    
    return qc

def decode(histogram):
    
    image = np.zeros((32, 32))
    for pxl in histogram:
        pos = int(bin(n)[:1:-1], 2)
        i = pos//32
        j = pos%32
        image[i][j] = np.arccos(np.sqrt(histogram.get(pxl, 0)/(histogram.get(pxl, 0) + histogram.get(pxl^1, 0))))*255*2/np.pi
        
    return image/2
    

def run_part1(image):
    #encode image into a circuit
    image = preprocess(image)
    
    circuit = encode(image)

    #simulate circuit
    histogram = simulate(circuit)
    print(histogram)
    
    #reconstruct the image
    image_re = decode(histogram)

    return circuit,image_re