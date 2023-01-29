#team: 5qbits
import cirq
import qiskit
import numpy as np
from sklearn.decomposition import PCA

n=len("data/images.npy")
mse=0
gatecount=0

from PIL import Image

#Encoder Function
def encode(image):
      
    imgmat = np.array(list(image.getdata(band=0)), float)
    imgmat.shape = (img.size[1], img.size[0])
    imgmat = np.matrix(imgmat)/255
        
       
  
    image = image.reshape((28, 28))
    pca = PCA(n_components=8)
    reduced_image = pca.fit_transform(image)
    reduced_image = reduced_image.reshape((8, 8))    

    def amplitude_encode(img_data):
        rms = np.sqrt(np.sum(np.sum(img_data**2, axis=1)))
    

        image_norm = []
        for arr in img_data:
            for ele in arr:
                image_norm.append(ele / rms)
        
# Return the normalized image as a numpy array
        return np.array(image_norm)

    image_norm_h = amplitude_encode(reduced_image)

    image_norm_v = amplitude_encode(reduced_image.T)  
    
    data_qb = 6
    anc_qb = 1
    total_qb = data_qb + anc_qb

    D2n_1 = np.roll(np.identity(2**total_qb), 1, axis=1)
 
    qc_h = QuantumCircuit(total_qb)
    qc_h.initialize(image_norm_h, range(1, total_qb))
    qc_h.h(0)
    qc_h.unitary(D2n_1, range(total_qb))
    qc_h.h(0)
    display(qc_h.draw('mpl', fold=-1))

    qc_v = QuantumCircuit(total_qb)
    qc_v.initialize(image_norm_v, range(1, total_qb))
    qc_v.h(0)
    qc_v.unitary(D2n_1, range(total_qb))
    qc_v.h(0)
    

    circ_list = [qc_h, qc_v]
        
    
    return qc_h

#Decoder Function
def decode(histogram):
    
    threshold = lambda amp: (amp > 1e-15 or amp < -1e-15)
    
    
    edge_scan_h = np.abs(np.array([1 if threshold(histogram.get(2*i+1).real) else 0 for i in range(2**6)])).reshape(8, 8)
    
    image = pca.inverse_transform(edge_scan_h)
    image= original_image.reshape((28, 28))
   
    return matrix

#Runner
def run_part1(image):
    #encode image into a circuit
    circuit=encode(image)

    #simulate circuit
    histogram=simulate(circuit)

    #reconstruct the image
    image_re=decode(histogram)

    return circuit,image_re

images[4]