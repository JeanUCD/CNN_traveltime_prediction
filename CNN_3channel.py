#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 21:26:55 2019

@author: jean
"""
''' import dataset'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# data loading and transforming
from torch.utils.data import DataLoader

import os
import sys

sys.path.append("..")
pwd = os.path.abspath('.') 

from GetThreeChannel import TTdataset,ToTensor
from torch.utils.data.sampler import SubsetRandomSampler
## Define a transform to read the data in as a tensor
data_transform = ToTensor()

'''
pred_interval : 'no_normal_5min': traveltime without normalization
                '5min': traveltime with Z-score normalization
                'uniform_5min': traveltime with uniform normalization to 0-1
'''
pred_interval = 'no_normal_5min'
csv_file1=pwd+'/TravelTime/{}/train_data/combined_flow.csv'.format(pred_interval)
csv_file2=pwd+'/TravelTime/{}/train_data/combined_occupancy.csv'.format(pred_interval)
csv_file3=pwd+'/TravelTime/{}/train_data/combined_observation.csv'.format(pred_interval)

test_csv_file1=pwd+'/TravelTime/{}/test_data/combined_flow.csv'.format(pred_interval)
test_csv_file2=pwd+'/TravelTime/{}/test_data/combined_occupancy.csv'.format(pred_interval)
test_csv_file3=pwd+'/TravelTime/{}/test_data/combined_observation.csv'.format(pred_interval)


root_dir1 = pwd+'/Pics/flow'
root_dir2 = pwd+'/Pics/occupancy'
root_dir3 = pwd+'/Pics/observation'
train_data = TTdataset(csv_file1, csv_file2, csv_file3, root_dir1, root_dir2, root_dir3, transform=data_transform)
test_data = TTdataset(test_csv_file1, test_csv_file2, test_csv_file3, root_dir1, root_dir2, root_dir3, transform=data_transform)

def create_datasets(batch_size, train_set, test_set):

    # percentage of training set to use as validation
    valid_size = 0.3

    # choose the training and test datasets
    train_data = train_set

    test_data =  test_set

    # obtain training indices that will be used for validation
    num_train = len(train_data)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]
    
    # define samplers for obtaining training and validation batches
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)
    
    # load training data in batches
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=train_sampler,
                                               num_workers=0)
    
    # load validation data in batches
    valid_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,
                                               sampler=valid_sampler,
                                               num_workers=0)
    
    # load test data in batches
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=batch_size,
                                              num_workers=0)
    
    return train_loader, test_loader, valid_loader
    
''' Define a CNN '''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # 3 input image channels (speed, flow, occupancy), 10 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F+2*P)/S +1 = (15-3)/1 +1 = 13
        # W: input width, F: kernel_size P: padding S: stride
        # the output Tensor for one image, will have the dimensions: (10, 13, 13)
        self.conv1 = nn.Conv2d(3, 10, 3)
        self.conv1_bn = nn.BatchNorm2d(10)
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # second conv layer: 10 inputs, 20 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (13-3)/1 +1 = 11
        # the output tensor will have dimensions: (20, 11, 11)
        # after another pool layer this becomes (20, 5, 5); 5.5 is rounded down
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.conv2_bn = nn.BatchNorm2d(20)
         # 20 outputs * the 5*5 filtered/pooled map size
        self.fc1 = nn.Linear(20*5*5, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        # finally, create 1 output channel 
        self.fc2 = nn.Linear(256, 128)
        self.fc2_bn = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc3_bn = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.fc4_bn = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32,16)
        self.fc5_bn = nn.BatchNorm1d(16)
        self.fc6 = nn.Linear(16,8)
        self.fc7 = nn.Linear(8,1)
        
        
        # dropout with p=0.5
        self.fc_drop = nn.Dropout(p=0.5)
        
    # define the feedforward behavior
    def forward(self, x):
        # two conv/relu + pool layers
        x = x.float()
        x = F.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = F.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.pool(x)
 
        # prep for linear layer
        # flatten the inputs into a vector
        x = x.view(x.size(0), -1)
        
        # 4 linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.fc_drop(x)
        #x = self.fc1_bn(x)
        
        x = F.relu(self.fc2(x))
        #x = self.fc_drop(x)
        #x = self.fc2_bn(x)
        x = F.relu(self.fc3(x))

        #x = self.fc3_bn(x)
        x = F.relu(self.fc4(x))

        #x = self.fc4_bn(x)
        x = F.relu(self.fc5(x))

        #x = self.fc5_bn(x)
        x = F.relu(self.fc6(x))
        x = self.fc7(x)
        # final output

        return x

   
# instantiate the Net
net = Net().float()

import torch.optim as optim
# stochastic gradient descent with a small learning rate
learning_rate = 0.002
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

# print it out!
from pytorchtools import EarlyStopping

''' Train the CNN'''
'''
Below, we've defined a `train` function that takes in a number of epochs to train for. 
* The number of epochs is how many times a network will cycle through the entire training dataset. 
* Inside the epoch loop, we loop over the training dataset in batches; recording the loss every 50 batches.

Here are the steps that this training function performs as it iterates over the training dataset:

1. Zero's the gradients to prepare for a forward pass
2. Passes the input through the network (forward pass)
3. Computes the loss (how far is the predicted classes are from the correct labels)
4. Propagates gradients back into the network’s parameters (backward pass)
5. Updates the weights (parameter update)
6. Prints out the calculated loss

'''

def train_model(model, batch_size, patience, n_epochs):
    
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []

    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 
    
    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    for epoch in range(1, n_epochs + 1):

        ###################
        # train the model #
        ###################
        model.train() # prep model for training
        for batch, data in enumerate(train_loader):
            inputs, labels = data
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(inputs)
            # calculate the loss
            outputs = outputs.reshape(-1)
            loss = criterion(outputs, labels)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # record training loss
            train_losses.append(loss.item())

        ######################    
        # validate the model #
        ######################
        model.eval() # prep model for evaluation
        for inputs, labels in valid_loader:
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(inputs)
            # calculate the loss
            outputs = outputs.reshape(-1)
            loss = criterion(outputs, labels)
            # record validation loss
            valid_losses.append(loss.item())

        # print training/validation statistics 
        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        
        epoch_len = len(str(n_epochs))
        
        print_msg = ("epoch:{}/{} \n".format(epoch,n_epochs)+
                     "train_loss: {}  ".format(train_loss)+
                     "valid_loss: {}".format(valid_loss))
        
        print(print_msg)
        
        # clear lists to track next epoch
        train_losses = []
        valid_losses = []
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        '''
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
        
    # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))
    '''
    torch.save(model.state_dict(), 'checkpoint.pt')
    return(model, avg_train_losses, avg_valid_losses)

batch_size = 20
n_epochs = 500

train_loader, test_loader, valid_loader = create_datasets(batch_size,train_data,test_data)

# early stopping patience; how long to wait after last time validation loss improved.
patience = 200

model, train_loss, valid_loss = train_model(net, batch_size, patience, n_epochs)



