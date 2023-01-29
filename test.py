import cirq
import numpy as np
import pickle
import json
import os
import sys
from collections import Counter
from sklearn.metrics import mean_squared_error

if len(sys.argv) > 1:
    data_path = sys.argv[1]
else:
    data_path = '.'

    
    
    
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# OpenMP: number of parallel threads.


# Plotting
import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

# Pennylane
import pennylane as qml
from pennylane import numpy as np

# Other tools
import time
import os
import copy

import numpy as np
import os
import random
import matplotlib.pyplot as plt

from skimage.io import imread, imsave
import skimage

# My code #

n_qubits = 4                     # Number of qubits
quantum = True                   # If set to "False", the dressed quantum circuit is replaced by 
                                 # An enterily classical net (defined by the next parameter). 
classical_model = '512_nq_2'     # Possible choices: '512_2','512_nq_2','551_512_2'. 
step = 0.0004                    # Learning rate
batch_size = 4                   # Number of samples for each training step
num_epochs = 30                  # Number of training epochs
q_depth = 6                      # Depth of the quantum circuit (number of variational layers)
gamma_lr_scheduler = 0.1         # Learning rate reduction applied every 10 epochs.                       
max_layers = 15                  # Keep 15 even if not all are used.
q_delta = 0.01                   # Initial spread of random quantum weights
rng_seed = 0                     # Seed for random number generator
start_time = time.time()         # Start of the computation timer

dev = qml.device('default.qubit', wires=n_qubits)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def H_layer(nqubits):
    """Layer of single-qubit Hadamard gates. 
    """
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)
        
def RY_layer(w):
    """Layer of parametrized qubit rotations around the y axis. 
    """
    for idx, element in enumerate(w):
        qml.RY(element, wires=idx)

def entangling_layer(nqubits):
    """Layer of CNOTs followed by another shifted layer of CNOT.
    """
    # In other words it should apply something like :
    #CNOT  CNOT  CNOT  CNOT...  CNOT
    #   CNOT  CNOT  CNOT...  CNOT  
    for i in range(0, nqubits - 1, 2): #loop over even indices: i=0,2,...N-2  
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2): #loop over odd indices:  i=1,3,...N-3
        qml.CNOT(wires=[i, i + 1])

# function to plot images
def imshow(inp, title=None):
    """Display image from tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    # We apply the inverse of the initial normalization operation.
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)

@qml.qnode(dev, interface='torch')
def q_net(q_in, q_weights_flat):
        
        # Reshape weights
        q_weights = q_weights_flat.reshape(max_layers, n_qubits)
        
        # Start from state |+> , unbiased w.r.t. |0> and |1>
        H_layer(n_qubits)   
        
        # Embed features in the quantum node
        RY_layer(q_in)      
       
        # Sequence of trainable variational layers
        for k in range(q_depth):
            entangling_layer(n_qubits)
            RY_layer(q_weights[k + 1])

        # Expectation values in the Z basis
        return [qml.expval(qml.PauliZ(j)) for j in range(n_qubits)]


class Quantumnet(nn.Module):
        def __init__(self):
            super().__init__()
            self.pre_net = nn.Linear(512, n_qubits)
            self.q_params = nn.Parameter(q_delta * torch.randn(max_layers * n_qubits))
            self.post_net = nn.Linear(n_qubits, 2)

        def forward(self, input_features):
            pre_out = self.pre_net(input_features) 
            q_in = torch.tanh(pre_out) * np.pi / 2.0   
            
            # Apply the quantum circuit to each element of the batch and append to q_out
            q_out = torch.Tensor(0, n_qubits)
            q_out = q_out.to(device)
            for elem in q_in:
                q_out_elem = q_net(elem,self.q_params).float().unsqueeze(0)
                q_out = torch.cat((q_out, q_out_elem))
            return self.post_net(q_out)

def train_model(model, criterion, optimizer, scheduler, num_epochs):
        since = time.time()
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_loss = 10000.0   # Large arbitrary number
        best_acc_train = 0.0
        best_loss_train = 10000.0  # Large arbitrary number
        print('Training started:')
        
        for epoch in range(num_epochs):    
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    scheduler.step()
                    # Set model to training mode
                    model.train()  
                else:
                    # Set model to evaluate mode
                    model.eval()
                running_loss = 0.0
                running_corrects = 0
                
                # Iterate over data.
                n_batches = dataset_sizes[phase] // batch_size
                it = 0
                for inputs, labels in dataloaders[phase]:
                    since_batch = time.time()
                    batch_size_ = len(inputs)
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    optimizer.zero_grad()
                    
                    # Track/compute gradient and make an optimization step only when training
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # Print iteration results
                    running_loss += loss.item() * batch_size_
                    batch_corrects = torch.sum(preds == labels.data).item()
                    running_corrects += batch_corrects
                    print('Phase: {} Epoch: {}/{} Iter: {}/{} Batch time: {:.4f}'.format(phase, epoch + 1, num_epochs, it + 1, n_batches + 1, time.time() - since_batch), end='\r', flush=True)
                    it += 1
                
                # Print epoch results
                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects / dataset_sizes[phase]
                print('Phase: {} Epoch: {}/{} Loss: {:.4f} Acc: {:.4f}        '.format('train' if phase == 'train' else 'val  ', epoch + 1, num_epochs, epoch_loss, epoch_acc))
                
                # Check if this is the best model wrt previous epochs
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
                if phase == 'train' and epoch_acc > best_acc_train:
                    best_acc_train = epoch_acc
                if phase == 'train' and epoch_loss < best_loss_train:
                    best_loss_train = epoch_loss
        
        # Print final results           
        model.load_state_dict(best_model_wts)
        time_elapsed = time.time() - since
        print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best test loss: {:.4f} | Best test accuracy: {:.4f}'.format(best_loss, best_acc))
        return model, best_acc
        


        
def run_part2_my():
    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(224),     # uncomment for data augmentation
            #transforms.RandomHorizontalFlip(),     # uncomment for data augmentation
            transforms.Resize(28),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # Normalize input channels using mean values and standard deviations of ImageNet.
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }


    # Load images and labels
    images = np.load("data/images.npy")
    labels = np.load("data/labels.npy")

    # Calculate number of images in each dataset
    num_images = len(images)
    num_train = int(0.8 * num_images)
    num_val = num_images - num_train

    # Split images and labels into train and val datasets
    train_images = images[:num_train]
    val_images = images[num_train:]
    train_labels = labels[:num_train]
    val_labels = labels[num_train:]

    # Make directories for train and val datasets
    os.makedirs("data/class/train", exist_ok=True)
    os.makedirs("data/class/val", exist_ok=True)

    # Make directories for True and False labels in train and val datasets
    os.makedirs("data/class/train/True", exist_ok=True)
    os.makedirs("data/class/train/False", exist_ok=True)
    os.makedirs("data/class/val/True", exist_ok=True)
    os.makedirs("data/class/val/False", exist_ok=True)

    # Save train images to corresponding True/False directories
    for i, (image, label) in enumerate(zip(train_images, train_labels)):
        plt.imsave(f"data/class/train/{label}/{i}.png", image, cmap='gray')

    # Save val images to corresponding True/False directories
    for i, (image, label) in enumerate(zip(val_images, val_labels)):
        plt.imsave(f"data/class/val/{label}/{i}.png", image, cmap='gray')

    data_transforms = {
        'train': transforms.Compose([
            #transforms.RandomResizedCrop(224),     # uncomment for data augmentation
            #transforms.RandomHorizontalFlip(),     # uncomment for data augmentation
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # Normalize input channels using mean values and standard deviations of ImageNet.
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    data_dir = 'data/class'
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                         data_transforms[x]) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # Initialize dataloader
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                      batch_size=batch_size, shuffle=True) for x in ['train', 'val']}
    
    
    

    # Get a batch of training data
    inputs, classes = next(iter(dataloaders['val']))
    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    #imshow(out, title=[class_names[x] for x in classes])

    torch.manual_seed(rng_seed)
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                      batch_size=batch_size,shuffle=True) for x in ['train', 'val']}



    model_hybrid = torchvision.models.resnet18(pretrained=True)

    for param in model_hybrid.parameters():
        param.requires_grad = False

    if quantum:
        model_hybrid.fc = Quantumnet()

    elif classical_model == '512_2':
        model_hybrid.fc = nn.Linear(512, 2)

    elif classical_model == '512_nq_2':
        model_hybrid.fc = nn.Sequential(nn.Linear(512, n_qubits), torch.nn.ReLU(), nn.Linear(n_qubits, 2)) 

    elif classical_model == '551_512_2':
        model_hybrid.fc = nn.Sequential(nn.Linear(512, 512), torch.nn.ReLU(), nn.Linear(512, 2))

    # Use CUDA or CPU according to the "device" object.
    model_hybrid = model_hybrid.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_hybrid = optim.Adam(model_hybrid.fc.parameters(), lr=step)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_hybrid, step_size=10, gamma=gamma_lr_scheduler)
    
    model_hybrid = train_model(model_hybrid, criterion, optimizer_hybrid,exp_lr_scheduler, num_epochs=2)
    
    return model_hybrid.best_acc

# My code #
        
#define utility functions



def count_gates(num_qubits, depth):
    """Returns the number of 1-qubit gates, number of 2-qubit gates, number of 3-qubit gates...."""
    
    counter = num_qubits * depth
        
    return counter


def image_mse(image1,image2):
    # Using sklearns mean squared error:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html
    return mean_squared_error(255*image1,255*image2)

def test():
    #load the actual hackthon data (fashion-mnist)
    images=np.load(data_path+'/images.npy')
    labels=np.load(data_path+'/labels.npy')
    
    #test part 1

    n=len(images)
    mse=0
    gatecount=0

    for image in images:
        #encode image into circuit
        circuit,image_re=run_part1(image)

        #count the number of 2qubit gates used
        gatecount+=count_gates(circuit)[2]

        #calculate mse
        mse+=image_mse(image,image_re)

    #fidelity of reconstruction
    f=1-mse/n
    gatecount=gatecount/n

    #score for part1
    score_part1=f*(0.999**gatecount)
    
    #test part 2
    
    #score
    score= run_part2_my()
    gatecount= count_gates(n_qubits, q_depth)

    score_part2=score*(0.999**gatecount)
    
    print(score_part1, ",", score_part2, ",", data_path, sep="")

############################
#      YOUR CODE HERE      #
############################
def encode(image):
    circuit=cirq.Circuit()
    if image[0][0]==0:
        circuit.append(cirq.rx(np.pi).on(cirq.LineQubit(0)))
    return circuit

def decode(histogram):
    if 1 in histogram.keys():
        image=np.array([[0,0],[0,0]])
    else:
        image=np.array([[1,1],[1,1]])
    return image

def run_part1(image):
    #encode image into a circuit
    circuit=encode(image)

    #simulate circuit
    histogram=simulate(circuit)

    #reconstruct the image
    image_re=decode(histogram)

    return circuit,image_re

def run_part2(image):
    # load the quantum classifier circuit
    with open('quantum_classifier.pickle', 'rb') as f:
        classifier=pickle.load(f)
    
    #encode image into circuit
    circuit=encode(image)
    
    #append with classifier circuit
    
    circuit.append(classifier)
    
    #simulate circuit
    histogram=simulate(circuit)
        
    #convert histogram to category
    label=histogram_to_category(histogram)
    
    #thresholding the label, any way you want
    if label>0.5:
        label=1
    else:
        label=0
        
    return circuit,label

############################
#      END YOUR CODE       #
############################

test()
