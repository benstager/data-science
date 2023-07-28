from os import path
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np

import sys
import warnings

if not sys.warnoptions:
    warnings.simplefilter('ignore')

"""
we seek to classify images using a multi-dimensional Softmax output, using 
a convolutional neural network
"""

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((.5,.5,.5),(.5,.5,.5))])

# gathering the data
trainset = torchvision.datasets.FashionMNIST(root= './fashionMNIST/', train=True, download=True, transform = transform)

testset = torchvision.datasets.FashionMNIST(root= './fashionMNIST/', train=False, download=True, transform = transform)


# hyperparms
batch_size = 100
n_iters = 5500
num_epochs = n_iters/(len(trainset)/batch_size)
num_epochs = int(num_epochs)

learning_rate = .001

# loading the data
train_loader = torch.utils.data.DataLoader(dataset=trainset, 
                                           batch_size=batch_size,            
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=testset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# softmax labels
classes = {0 : 'T-Shirt/Top', 1 : 'Trouser', 2 : 'Pullover', 3 : 'Dress',
           4 : 'Coat', 5 : 'Sandal', 6 : 'Shirt', 7 : 'Sneaker', 8 : 'Bag',
           9 : 'Ankle boot'}

# plotting some images
def imshow(image, label):
    plt.title(classes[label])
    plt.imshow(image.reshape(28, 28), cmap = 'Greys', interpolation = 'nearest')

fig = plt.figure(figsize=(10,10))
rows = 4
columns = 5

for num in range(0,20):
    image, label = trainset.train_data[num], trainset.train_labels[num].item()
    num += 1
    fig.add_subplot(rows, columns, num)
    imshow(image, label)

plt.show()


"""
now we can start to write the model, with forward flow:
input->conv1->batch norm->ReLu->max pool1->conv2->bach norm->ReLu->max pool2->FC->softmax
"""
class convnet1(nn.Module):
    def __init__(self):
        super(convnet1, self).__init__()
        
        # Constraints for layer 1
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride = 1, padding=2)
        self.batch1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2) #default stride is equivalent to the kernel_size
        
        # Constraints for layer 2
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride = 1, padding=2)
        self.batch2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Defining the Linear layer
        self.fc = nn.Linear(32*7*7, 10)
    
    # defining the network flow
    def forward(self, x):
        # Conv 1
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.relu1(out)
        
        # Max Pool 1
        out = self.pool1(out)
        
        # Conv 2
        out = self.conv2(out)
        out = self.batch2(out)
        out = self.relu2(out)
        
        # Max Pool 2
        out = self.pool2(out)
        
        out = out.view(out.size(0), -1)
        # Linear Layer
        out = self.fc(out)
        
        return out
    
# creating an object from convnet1
model = convnet1()
model.parameters

# hyperparms
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)


# train model


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader, 0):
        images = Variable(images.float())
        labels = Variable(labels)
        
        # Forward + Backward + Optimizer
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
