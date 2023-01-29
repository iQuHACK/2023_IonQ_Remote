team_name = "missing_cats"


import numpy as np
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, assemble, Aer, transpile
from qiskit.visualization import plot_histogram


image_data=np.load('data/images.npy')
label_data=np.load('data/labels.npy')

image = image_data[14]
def encode(image):


  # Initialize the quantum circuit for the image 
  # Pixel position
  idx = QuantumRegister(56, 'idx')
  # grayscale pixel intensity value
  intensity = QuantumRegister(8,'intensity')
  # classical register
  cr = ClassicalRegister(64, 'cr')

  # create the quantum circuit for the image
  qc_image = QuantumCircuit(intensity, idx, cr)

  # set the total number of qubits
  num_qubits = qc_image.num_qubits






  # Initialize the quantum circuit

  # Optional: Add Identity gates to the intensity values
  for idx in range(intensity.size):
    qc_image.i(idx)

  # Add Hadamard gates to the pixel positions
  for x in range(8,36):

    qc_image.h(x)

  # Separate with barrier so it is easy to read later.
  qc_image.barrier()

  xv = []
  for i_index ,i in enumerate(image):
    i = image[14]

    for j_index, j in enumerate(i):
    #converting the pixel value to binary
      x = int(j*255*255) # 256 or 255 what if a pixel value is 1 then does it go out fo index or something? 
      
      xv.append((j,x))
      x = str(bin(x)[2:])
      
      check = []
      

      if int(x)== 0:
        for idx in range(num_qubits):
          qc_image.i(idx)
        
        qc_image.barrier()
        continue
      

      for idx , px_value in enumerate(x[::-1]):
        check.append(px_value)
        if(px_value=='1'):
          qc_image.ccx(8+i_index, 36+j_index, idx) # what should be placed here instead of num_qubits-1 and num_qubits-2

      # Reset the NOT gate
      qc_image.x(num_qubits-1) #place a not and remove a not here ok ? tf  how do we do thi sfo r28 by fukin 28 ?
      qc_image.barrier()
      print(qc_image.draw())
      print("Pixel Value",x,check)
      take = input() 

        
    print(qc_image.draw())
    print("Pixel Value",x,check,xv)
    return
      

  qc_image.measure(range(36),range(36))


  print('Circuit dimensions')
  print('Circuit depth: ', qc_image.decompose().depth())
  print('Circuit size: ', qc_image.decompose().size())

  qc_image.decompose().count_ops()


  aer_sim = Aer.get_backend('aer_simulator')
  t_qc_image = transpile(qc_image, aer_sim)
  qobj = assemble(t_qc_image, shots=8192)
  job_neqr = aer_sim.run(qobj)
  result_neqr = job_neqr.result()
  counts_neqr = result_neqr.get_counts()
  
  plot_histogram(counts_neqr)

encode(image)

