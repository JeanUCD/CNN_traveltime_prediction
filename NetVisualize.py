# -*- coding: utf-8 -*-
"""
Created on Tue May 28 22:33:12 2019

@author: jean
"""

#!/usr/bin/env python
# coding: utf-8

# # CNN for Classification
# ---
# In this notebook, we define **and train** an CNN to classify images from the [Fashion-MNIST database](https://github.com/zalandoresearch/fashion-mnist).

# ### Load the [data](http://pytorch.org/docs/master/torchvision/datasets.html)
# 
# In this cell, we load in both **training and test** datasets from the FashionMNIST class.



# our basic libraries
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

test_csv_file1=pwd+'/TravelTime/test_data/1014/speed.csv'
test_csv_file2=pwd+'/TravelTime/test_data/1014/flow.csv'
test_csv_file3=pwd+'/TravelTime/test_data/1014/occupancy.csv'
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
    

