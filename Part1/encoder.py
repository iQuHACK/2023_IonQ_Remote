import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, QuantumRegister


class Encoder:
    def __init__(self, images, n_resize=(4, 4)) -> None:
         self.images = images
         self.images_resized = np.resize(images, (len(images), *n_resize))
         self.images_normalized = self.images_resized/np.max(np.abs(self.images_resized))
    
    def encoder(self, image:np.ndarray):
        info_image = np.ndarray.flatten(image)
        n_qubits = len(info_image)
        q_register = QuantumRegister(n_qubits)
        qc = QuantumCircuit(q_register, name='Encoder')

        for i, pixel_value in enumerate(info_image):
                qc.rx(pixel_value, q_register[i])
        return qc