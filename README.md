<b><h1>Welcome to Schrödinger's Descendants - MIT iQuHACK 2023</h1></b>
![Uploading image.png…]()

<img src="E:/Quantum Computing/Resources and Content/MIT.jpg" width="1000" height="350"/>


This project is a part of the MIT iQuHACK2023. It is an annual quantum hackathon. It aims to bring students (high school through early-career professionals) from a diverse set of backgrounds to explore improvements and applications of near-term quantum devices. MIT iQuHACK 2023 will have an in-person hackathon and a virtual hackathon. Our team will take part in the remote hackathon. We have chosen the IonQ Challenge. # MIT_iQuHack-2023

***Our motto**: Never too infeasible to be impossible!*


<h3>These are our teammates(listed alphabetically):</h3>

1. [Akash Reddy](https://github.com/Akash6300)

2. [Gayatri Vadaparty](https://github.com/GayatriVadaparty)

3. [Kiran Kaur](https://github.com/KyranKaur)

4. [Nanda Kishore Reddy Aavu](https://github.com/nandakishore1807/)

5. [Sai Ganesh Manda](https://github.com/mvsg2)

<h1>IonQ Challenge - Remote</h1>

Our team decided to choose the IonQ challenge from the 2 challenges available for the remote participants this year. You can find the challenge GitHub Repository [here](https://github.com/iQuHACK/2023_IonQ_Remote) and the specific challenge [here](https://github.com/iQuHACK/2023_IonQ_Remote/blob/main/MIT%20iQuHACK%20remote%20challenge%20.docx).

<h1>Quantum Image Processing</h1>

Image processing is extensively used in fast growing markets like facial recognition and autonomous vehicles. At the same time Quantum Image Processing is an emerging field of Quantum Information Science that holds the promise of considerable speed-up for specific but commonly used operations like edge detection. For example, Zhang et al. proposed in 2014 a novel quantum image edge extraction algorithm (QSobel) based on the Flexible Representation of Quantum Images (FRQI) representation and the classical edge extraction algorithm Sobel. QSobel can extract edges in the computational complexity of O(n^2) for a FRQI quantum image with a size of  2^n×2^n, which is a significant and exponential speedup compared with existing edge extraction algorithms. We used two methods of encoding images in quantum states, the Flexible Representation of Quantum Images (FRQI) and the Novel Enhanced Quantum Representation (NEQR). Once our image is encoded in these states, we can then process them using other quantum algorithms.

<h1>Part1 - Data Loading</h1>


The first step is to encode images captured by camera into quantum circuits. This way the quantum computer can “see” the item. We were given an image dataset (Fashion-MNIST) and our task was to make a data loading scheme that encodes the images into a quantum state as lossless as possible. The encoded image had to be interpretable by simple measurements at the end of the circuit. 

The given dataset is in the so-called binary format for numpy objects or pickled files. In order to load it, use the NumPy load function which will automatically load these objects from the disk and will return a numpy array of data stored in the file. (Refer to the documentation page linked below for more information).

Usage:  *np.load(...# arguments # ...)*    # having imported the NumPy library with the command: _import numpy as np_


<h1>Encoding and Decoding Qubits</h1>

Data representation is crucial for the success of machine learning models. For classical machine learning, the problem is how to represent the data numerically, so that it can be best processed by a classical machine learning algorithm.

For quantum machine learning, this question is similar, but more fundamental: how to represent and efficiently input the data into a quantum system, so that it can be processed by a quantum machine learning algorithm? This is usually referred to as data encoding. This process is a critical part of quantum machine learning algorithms and directly affects their computational power.

We first encoded the preprocessed image dataset into quantum circuits and then ran it through quantum machine learning algorithms. After measurement of the circuits we represented the data in the form of histograms which were decode back to images.

<h1>Part2 - Classification</h1>

In this part of the project we classified our images according to the appropriate labels. We used encoder (image) to convert the image into a quantum circuit,append the circuit with the classifier circuit loaded from the .pickle file. Then we simulated the circuit (encoded_image+classifier_circuit) to get a histogram and ran the provided histogram_to_label (histogram) to convert the histogram to label.

<h1><b>Resources and References</b></h1>

[NumPy load documentation](https://numpy.org/doc/stable/reference/generated/numpy.load.html)

[Quantum Image Processing chapter in Qiskit textbook](https://qiskit.org/textbook/ch-applications/image-processing-frqi-neqr.html)

[Image Processing on Quantum Computers](https://paperswithcode.com/paper/image-processing-in-quantum-computers/review/)

[NEQR Paper](https://www.researchgate.net/publication/257641933_NEQR_A_novel_enhanced_quantum_representation_of_digital_images#pf10)

[Data Encoding Qiskit](https://learn.qiskit.org/course/machine-learning/data-encoding)

[Simple Scheme for Encoding and Decoding a Qubit in an Unknown State](https://www.nature.com/articles/srep08975)

[Quantum Image Processing](https://arxiv.org/ftp/arxiv/papers/2002/2002.04394.pdf)

https://docs.google.com/document/d/125NR-4mMqrQg4Q_p09pxOP91K8kcQAs_SRR0YeSR1M8/edit

https://docs.google.com/document/d/1jDr47urVEKvUAX0gAKCfhUjR3tRtDZIxw3WdoilN4JY/edit
