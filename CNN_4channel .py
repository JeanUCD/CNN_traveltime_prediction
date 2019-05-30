#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 21:26:55 2019

@author: jean
"""
''' import dataset'''
# our basic libraries
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


# data loading and transforming 
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

import os
import sys

sys.path.append("..")
pwd = os.path.abspath('.') 

from GetChannel import TTdataset,ToTensor

## Define a transform to read the data in as a tensor
data_transform = ToTensor()

# choose the training and test datasets
csv_file1=pwd+'/TravelTime/combined_speed.csv'
csv_file2=pwd+'/TravelTime/combined_flow.csv'
csv_file3=pwd+'/TravelTime/combined_occupancy.csv'
csv_file4=pwd+'/TravelTime/combined_observation.csv'

test_csv_file1=pwd+'/TravelTime/test_data/1105/speed.csv'
test_csv_file2=pwd+'/TravelTime/test_data/1105/flow.csv'
test_csv_file3=pwd+'/TravelTime/test_data/1105/occupancy.csv'
test_csv_file4=pwd+'/TravelTime/test_data/1105/observation.csv'


root_dir1 = pwd+'/Pics/speed'
root_dir2 = pwd+'/Pics/flow'
root_dir3 = pwd+'/Pics/occupancy'
root_dir4 = pwd+'/Pics/observation'

train_data = TTdataset(csv_file1, csv_file2, csv_file3, csv_file4, root_dir1, root_dir2, root_dir3,root_dir4, transform=data_transform)
test_data = TTdataset(test_csv_file1, test_csv_file2, test_csv_file3,test_csv_file4, root_dir1, root_dir2, root_dir3,root_dir4, transform=data_transform)

# Print out some stats about the training and test data
print('Train data, number of images: ', len(train_data))
print('test data, number of images: ', len(test_data))

# prepare data loaders, set the batch_size
## when you get to training your network, see how batch_size affects the loss
batch_size = 25

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)


''' Define a CNN '''
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        # 1 input image channel (grayscale), 10 output channels/feature maps
        # 3x3 square convolution kernel
        ## output size = (W-F+2*P)/S +1 = (15-3)/1 +1 = 13
        # W: input width, F: kernel_size P: padding S: stride
        # the output Tensor for one image, will have the dimensions: (10, 13, 13)
        # after one pool layer, this becomes (10, 13, 13)
        self.conv1 = nn.Conv2d(4, 10, 3)
        
        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # second conv layer: 10 inputs, 20 outputs, 3x3 conv
        ## output size = (W-F)/S +1 = (13-3)/1 +1 = 11
        # the output tensor will have dimensions: (20, 11, 11)
        # after another pool layer this becomes (20, 5, 5); 5.5 is rounded down
        self.conv2 = nn.Conv2d(10, 20, 3)
        
         # 20 outputs * the 5*5 filtered/pooled map size
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


def AccuTest(model,test_loader,criterion):
    '''Return the loss on test set and the prediction accuracy'''
    correct = 0
    total = 0
    accuracy = 0
    # initialize tensor and lists to monitor test loss and accuracy
    test_loss = torch.zeros(1)

    # set the module to evaluation mode
    model.eval()

    for batch_i, data in enumerate(test_loader):
    
        # get the input images and their corresponding labels
        inputs, labels = data
    
        # forward pass to get outputs
        outputs = model(inputs)

        # calculate the loss
        loss = criterion(outputs, labels.long())
            
        # update average test loss 
        test_loss = test_loss + ((torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss))
    
        # get the predicted class from the maximum value in the output-list of class scores
        _, predicted = torch.max(outputs.data, 1)

        # count up total number of correct lTensorabels
        # for which the predicted and true labels are equal
        total += labels.size(0)
        #correct += ((predicted == labels.long()).sum() + 
                    #(predicted == (labels.long()+1)).sum() + (predicted == (labels.long()-1)).sum())
        correct += (predicted == labels.long()).sum()
    # to convert `correct` from a  into a scalar, use .item()
    accuracy = 100.0 * correct.item() / total
    test_loss = test_loss.numpy()[0]
    
    return test_loss, accuracy
    
# instantiate and print your Net
net = Net().float()

import torch.optim as optim
# stochastic gradient descent with a small learning rate
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
# show the accuracy before training
_,accuracy0 = AccuTest(net,train_loader,criterion)

# print it out!
print('Accuracy before training: {}%'.format(accuracy0))

''' Train the CNN'''
'''
Below, we've defined a `train` function that takes in a number of epochs to train for. 
* The number of epochs is how many times a network will cycle through the entire training dataset. 
* Inside the epoch loop, we loop over the training dataset in batches; recording the loss every 1000 batches.

Here are the steps that this training function performs as it iterates over the training dataset:

1. Zero's the gradients to prepare for a forward pass
2. Passes the input through the network (forward pass)
3. Computes the loss (how far is the predicted classes are from the correct labels)
4. Propagates gradients back into the network’s parameters (backward pass)
5. Updates the weights (parameter update)
6. Prints out the calculated loss

'''

def train(n_epochs, model):
    
    loss_over_time = [] # to track the loss as the network trains
    test_loss = []
    accuracy = []
    accuracy.append(accuracy0)
    # switch to the training model
    model.train()
    
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        running_loss = 0.0
        
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            inputs, labels = data
            
            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # forward pass to get outputs
            outputs = model(inputs)
           
            # calculate the loss
            loss = criterion(outputs, labels.long())

            # backward pass to calculate the parameter gradients
            loss.backward()

            # update the parameters
            optimizer.step()

            # print loss statistics
            # to convert loss into a scalar and add it to running_loss, we use .item()
            running_loss += loss.item()
            
            if batch_i % 50 == 49: # print every 50 batches
                avg_loss = running_loss/50
                # record and print the avg loss over the 50 batches
                loss_over_time.append(avg_loss)
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, avg_loss))
                running_loss = 0.0
                
        test_loss_epoch,accuracy_epoch = AccuTest(model,test_loader,criterion)
        
        test_loss.append(test_loss_epoch)
        accuracy.append(accuracy_epoch)
    print('Finished Training')
    return loss_over_time,test_loss,accuracy

# define the number of epochs to train for
n_epochs = 20 # start small to see if your model works, initially

# call train and record the loss over time
training_loss, test_loss, accuracy_on_testset = train(n_epochs,net)

# visualize the loss as the network trained
plt.plot(training_loss)
plt.xlabel('50\'s of batches')
plt.ylabel('loss')
#plt.ylim(0, 100000) # consistent scale
plt.show()





