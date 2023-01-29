{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3a8f1303-c991-4af4-9cad-649ae271c00f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(28, 28)\n"
     ]
    }
   ],
   "source": [
    "import numpy as np\n",
    "import qiskit\n",
    "\n",
    "# Encoder function to convert an image into a quantum circuit\n",
    "def encoder(image):\n",
    "    height, width = image.shape\n",
    "    print(image.shape)\n",
    "    qc = qiskit.QuantumCircuit(height * width, height * width)\n",
    "\n",
    "    # Apply Hadamard gates to the qubits\n",
    "    for i in range(height * width):\n",
    "        qc.h(i)\n",
    "\n",
    "    # Apply the NEQR encoding\n",
    "    for row in range(height):\n",
    "        for col in range(width):\n",
    "            # Get the pixel value at the current position\n",
    "            pixel_value = image[row][col]\n",
    "\n",
    "            # Apply controlled-Z gates based on the pixel value\n",
    "            for i in range(height * width):\n",
    "                if (i // width == row and i % width == col):\n",
    "                    if pixel_value == 0:\n",
    "                        qc.z(i)\n",
    "                else:\n",
    "                    qc.cz(i, (row * width) + col)\n",
    "\n",
    "    return qc\n",
    "\n",
    "# Simulator function to simulate a circuit and get a histogram\n",
    "def simulator(circuit):\n",
    "    # Simulate the circuit using the statevector simulator\n",
    "    backend = qiskit.Aer.get_backend('statevector_simulator')\n",
    "    result = qiskit.execute(circuit, backend).result()\n",
    "    statevector = result.get_statevector()\n",
    "\n",
    "    # Get the probability distribution of the statevector\n",
    "    probabilities = np.abs(statevector)**2\n",
    "\n",
    "    return probabilities\n",
    "\n",
    "# Decoder function to convert the histogram into a regenerated image\n",
    "def decoder(histogram, image):\n",
    "    # Reshape the histogram into an image\n",
    "    height, width = image.shape\n",
    "    image = np.zeros((height, width))\n",
    "\n",
    "    for i in range(height * width):\n",
    "        row = i // width\n",
    "        col = i % width\n",
    "        image[row][col] = histogram[i]\n",
    "\n",
    "    return image\n",
    "\n",
    "# Example usage\n",
    "if __name__ == \"__main__\":\n",
    "    # Load the image\n",
    "    data_path='data'\n",
    "#load the actual hackthon data (fashion-mnist)\n",
    "    image=np.load(data_path+'/images.npy')\n",
    "    #image = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])\n",
    "\n",
    "    # Convert the image into a quantum circuit\n",
    "    circuit = encoder(image[1])\n",
    "\n",
    "    # Simulate the circuit and get a histogram\n",
    "    histogram = simulator(circuit)\n",
    "\n",
    "    # Convert the histogram into a regenerated image\n",
    "    regenerated_image = decoder(histogram, image[1])\n",
    "\n",
    "    print(\"Original Image:\")\n",
    "    print(image)\n",
    "    print(histogram)\n",
    "    print(\"Regenerated Image:\")\n",
    "    print(regenerated_image)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b5acf433-e1d7-4a07-81c7-355ef73c7aaf",
   "metadata": {},
   "outputs": [],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.imshow(image[1])\n",
    "plt.imshow(regenerated_image)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "a100e8c4-d54d-498d-ae02-9b17636cecbf",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 [IonQ]",
   "language": "python",
   "name": "python3_ionq_6vdluz"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
