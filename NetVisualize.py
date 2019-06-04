# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:33:12 2019

@author: jean
"""
''' have a look of the images and labels'''
# import basic libraries
import torch
import torchvision

# data loading and transforming
from torch.utils.data import DataLoader
from torchvision import transforms
import os
import sys

sys.path.append("..")
pwd = os.path.abspath('.') 

from GetThreeChannel import TTdataset,ToTensor
# The output of torchvision datasets are PILImage images of range [0, 1]. 
# We transform them to Tensors for input into a CNN

## Define a transform to read the data in as a tensor
data_transform = ToTensor()

# choose the training and test datasets
csv_file1=pwd+'/TravelTime/combined_speed.csv'
csv_file2=pwd+'/TravelTime/combined_flow.csv'
csv_file3=pwd+'/TravelTime/combined_occupancy.csv'
root_dir1 = pwd+'/Pics/speed'
root_dir2 = pwd+'/Pics/flow'
root_dir3 = pwd+'/Pics/occupancy'
train_data = TTdataset(csv_file1, csv_file2, csv_file3, root_dir1, root_dir2, root_dir3, transform=data_transform)

test_csv_file1=pwd+'/TravelTime/test_data/1031/speed.csv'
test_csv_file2=pwd+'/TravelTime/test_data/1031/flow.csv'
test_csv_file3=pwd+'/TravelTime/test_data/1031/occupancy.csv'
test_data = TTdataset(test_csv_file1, test_csv_file2, test_csv_file3, root_dir1, root_dir2, root_dir3, transform=data_transform)
# Print out some stats about the training and test data
print('Train data, number of images: ', len(train_data))
print('Test data, number of images: ', len(test_data))


# prepare data loaders, set the batch_size
batch_size = 10

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

# ### Visualize some training data
# 
# This cell iterates over the training dataset, loading a random batch of image/label data, using `dataiter.next()`. It then plots the batch of images and labels in a `2 x batch_size/2` grid.


import numpy as np
import matplotlib.pyplot as plt

    
# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()[:,0]

# plot the images in the batch, along with the corresponding labels
fig = plt.figure()
for idx in np.arange(batch_size):
    ax = fig.add_subplot(2, batch_size/2, idx+1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(images[idx]), cmap='gray')
    ax.set_title('{}'.format(labels[idx]))
    
    
''' Now start to visualize the CNN'''
import torch.nn as nn
import torch.nn.functional as F
''' Define a CNN '''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # 3 input image channels (speed,flow,occupancy), 10 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F+2*P)/S +1 = (15-3)/1 +1 = 13
        # W: input width, F: kernel_size P: padding S: stride
        # the output Tensor for one image, will have the dimensions: (10, 13, 13)
        # after one pool layer, this becomes (10, 13, 13)
        self.conv1 = nn.Conv2d(3, 10, 3)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # second conv layer: 10 inputs, 20 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (13-3)/1 +1 = 11
        # the output tensor will have dimensions: (20, 11, 11)
        # after another pool layer this becomes (20, 5, 5); 5.5 is rounded down
        self.conv2 = nn.Conv2d(10, 20, 3)
        
         # 50 outputs * the 5*5 filtered/pooled map size
        self.fc1 = nn.Linear(20*5*5, 50)
        
        # finally, create 1 output channel 
        self.fc2 = nn.Linear(50, 10)
        
        # dropout with p=0.5
        self.fc1_drop = nn.Dropout(p=0.5)
        
        
        

    # define the feedforward behavior
    def forward(self, x):
        # two conv/relu + pool layers
        x = x.float()
        x = (F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))

        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
        
        # two linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        x = self.fc2(x)

        # final output
                  
        return x

net = Net()
print(net)
"""
Net(
  (conv1): Conv2d(3, 10, kernel_size=(3, 3), stride=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(10, 20, kernel_size=(3, 3), stride=(1, 1))
  (fc1): Linear(in_features=500, out_features=50, bias=True)
  (fc2): Linear(in_features=50, out_features=10, bias=True)
  (fc1_drop): Dropout(p=0.5)
)
"""

