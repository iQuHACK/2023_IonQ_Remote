

Machine learning is a technology that has attracted a great deal of attention due to its high performance and versatility. In fact, it has been put to practical use in many industries with the recent development of algorithms and the increase of computational resources. A typical example is computer vision, where machine learning is now able to classify images with the same or better accuracy than humans. For example, the ability to automatically classify clothing images has made online shopping for clothes more convenient.
The application of quantum computation to machine learning has recently been shown to have the potential for even greater capabilities. Various algorithms have been proposed for quantum machine learning, such as the quantum support vector machine (QSVM) and quantum generative adversarial networks (QGANs). In this challenge, you will use QSVM to tackle the clothing image classification task.
QSVM is a quantum version of the support vector machine (SVM), a classical machine learning algorithm. There are various approaches to QSVM, some aim to accelerate computation assuming fault-tolerant quantum computers, while others aim to achieve higher expressive power assuming noisy, near-term devices. In this challenge, we will focus on the latter, and the details will be explained later.
For this implementation of QSVM, you will be able to make choices on how you want to compose your quantum model, in particular focusing on the quantum feature map. 

This challenge is basically binary classification problem where we need to classify t-shirts and non t-shirts. We have followed the following steps to solve this challenge.
* QSVM for binary classification of fashion-MNIST data
* Quantum Feature Mapping
* Quantum Kernel Estimation
* Quantum Support Vector Machine (QSVM) preparation
* Accuracy test
