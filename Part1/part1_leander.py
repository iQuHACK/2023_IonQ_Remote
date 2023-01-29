import numpy as np
from qiskit import ClassicalRegister, QuantumRegister, QuantumCircuit, Aer, transpile
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
from functools import partial
from multiprocessing import Process, Manager
from scipy.signal import convolve2d


def execute_circuit(qc, shots=1024, device=None):
    device = Aer.get_backend('qasm_simulator') if device is None else device
    transpiled_circuit = transpile(qc, device)
    counts = device.run(transpiled_circuit, shots=shots).result().get_counts()
    return counts

def basis_states_probs(counts):
    n = len(list(counts.keys())[0])
    N = sum(list(counts.values()))
    return np.array([counts[np.binary_repr(vals,n)]/N if counts.get(np.binary_repr(vals,n)) is not None else 0 for vals in range(2**n)])


class Encoder:
    def __init__(self, images, n_process=1, n_decode=2):
        self.images = images
        self.max_factor = np.max(images)
        self.images = images/self.max_factor
        self.images_shape = self.images[0].shape
        self.kernel = np.random.random((7, 7))
        print(self.kernel)
        self.n_decode = n_decode
        self.d = 7
        self.decode_kernels = [np.random.random((self.d, self.d)) for _ in range(self.n_decode)]
        self.kernel_shape = self.kernel.shape
        self.N_qubits = 16
        self.n_process = n_process
    
    def reduze_size(self, image):
        Nx, Ny = self.images_shape
        kx, ky = self.kernel_shape
        Dx = Nx//kx
        Dy = Ny//ky
        new_image = np.zeros((Dx, Dy))
        for x in range(Dx):
            for y in range(Dy):
                new_image[x, y] = np.sum(self.kernel * image[kx*x: kx*(x +1), ky*y: ky*(y+1)])
        new_image = new_image/np.max(new_image)*np.max(image)
        return new_image
    
    def encode(self, image):
        image_reduzed = self.reduze_size(image)
        q_register = QuantumRegister(self.N_qubits)
        c_register = ClassicalRegister(self.N_qubits)
        qc = QuantumCircuit(q_register, c_register)
        
        data = np.ndarray.flatten(image_reduzed) * np.pi
        for i, d in enumerate(data):
            qc.ry(d, q_register[i])
            qc.measure(q_register[i], c_register[i])
        
        return qc
    
    def apply_decoder(self, kernel, image_resized):
        
        Nx, Ny = image_resized.shape
        kx, ky = kernel.shape
        Dx = Nx*kx
        Dy = Ny*ky
        new_image = np.zeros((Dx, Dy))
        for x in range(Nx):
            for y in range(Ny):
                new_image[kx*x: kx*(x +1), ky*y: ky*(y+1)] = kernel * image_resized[x, y]
        return new_image
        

    def decoder(self, probs):
        #q_probs = []
        #for q in range(1, self.N_qubits+1):
        #    q_probs.append(np.sum(np.array(np.split(probs,2**q))[np.arange(0,2**q,2)]))
        #q_probs = np.array(q_probs)[::-1]
        #data = np.arccos(np.sqrt(q_probs))*2/np.pi
        data = probs
        image_decoded = np.resize(data, (4, 4))
        result = image_decoded.copy()
        result = self.apply_decoder(self.decode_kernels[0], result)
        def layer(kernel, M):
            res = convolve2d(kernel, np.arctan(M))
            nx, ny = res.shape
            res = res[nx//2-14:nx//2+14, ny//2-14:ny//2+14]
            return res 

        for kernel in self.decode_kernels[1:]:
            result += layer(kernel, result)
        result /= np.max(result) * np.max(image_decoded)
        nx, ny = result.shape
        result = result[nx//2-14:nx//2+14, ny//2-14:ny//2+14]
        return result
    
    def loss(self, image):
        #qc_encoder = self.encode(image)
        #probs = basis_states_probs(execute_circuit(qc_encoder))
        probs = np.ndarray.flatten(self.reduze_size(image))
        image_res = self.decoder(probs)
        return mean_squared_error(image, image_res)

    
    def optimize(self):
        self.epoch = 0
        self.BestLoss = 10
        def f(kernels):
            kernel, decode_kernel = kernels[:7*7], kernels[7*7:]
            self.kernel = np.reshape(kernel, (7, 7))
            self.decode_kernels = np.reshape(decode_kernel, (self.n_decode, self.d, self.d))
            L = 0
            N = len(self.images)
            def loss(images):
                N = len(images)
                L = []
                for i, image in enumerate(images):
                    l = self.loss(image)
                    #print(f'{i}/{N} Loss: {l}', end='\r')
                    L.append(l)
                return L
            images = self.images.copy()
            np.random.shuffle(images)
            loss_vals = self.parallelize('Compute Loss', loss, images[:1000])
            L = np.mean(loss_vals)
            print(f'Epoch: {self.epoch} Loss: {L}')
            if L < self.BestLoss:
                np.save(open('kernels.npy', 'wb'), self.kernel)
                print('Kernels: ', kernels)
                self.BestLoss = L
            self.epoch += 1
            return L

        kernels = np.random.random(self.d*self.d*self.n_decode + 7*7)
        print(kernels.shape)
        self.res = minimize(f, kernels)
        np.save(open('res.npy', 'wb'),self.res.x)
        print(self.res.x)
    
    def parallelize(self, process_name: str, f, iterator, *args) -> np.ndarray:
       
        process = []
        iterator = list(iterator)
        N = len(iterator)

        def parallel_f(result, per, iterator, *args) -> None:
            '''
            Auxiliar function to help the parallelization

            Parameters:
                result : array_like
                    It is a shared memory list where each result is stored.
                per : list[int]
                    It is a shared memory list that contais the number of elements solved.
                iterator : array_like
                    The function f is applied to elements in the iterator array.
            '''
            value = f(iterator, *args)              # The function f is applied to the iterator
            if value is not None:
                # The function may not return anything
                result += f(iterator, *args)        # Store the output into result array
            per[0] += len(iterator)                 # The counter is actualized
            #print(per[0]/N, end='\r')
        
        result = Manager().list([])             # Shared Memory list to store the result
        per = Manager().list([0])               # Shared Memory to countability the progress
        f_ = partial(parallel_f,  result, per)  # Modified function used to create processes

        n = N//self.n_process                                                   # Number or processes
        for i_start in range(self.n_process):
            # Division of the iterator array into n smaller arrays
            j_end = n*(i_start+1) if i_start < self.n_process-1\
                else n*(i_start+1) + N % self.n_process
            i_start = i_start*n
            p = Process(target=f_, args=(iterator[i_start: j_end], *args)) 
            #print(f'Create Proces: {i_start}')     # Process creation
            p.start()                                                           # Initialize the process
            process.append(p)

        while len(process) > 0:
            p = process.pop(0)
            p.join()

        return np.array(result)


with open('../data/images.npy', 'rb') as f:
    images = np.load(f)
Training = Encoder(images)
#Training.optimize()

    
