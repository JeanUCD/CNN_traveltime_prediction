# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:13:30 2019

@author: jean
"""
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# data loading and transforming
from torch.utils.data import DataLoader
import numpy as np
from tsne import bh_sne

import os
import sys

sys.path.append("..")
pwd = os.path.abspath('.') 

from GetThreeChannel import TTdataset,ToTensor
data_transform = None

csv_file1=pwd+'/TravelTime/combined_speed.csv'
csv_file2=pwd+'/TravelTime/combined_flow.csv'
csv_file3=pwd+'/TravelTime/combined_occupancy.csv'
root_dir1 = pwd+'/Pics/speed'
root_dir2 = pwd+'/Pics/flow'
root_dir3 = pwd+'/Pics/occupancy'
train_data = TTdataset(csv_file1, csv_file2, csv_file3, root_dir1, root_dir2, root_dir3, transform=data_transform)

X = []
Y = []
for i in range(len(train_data)):
    X.append(train_data.__getitem__(i)['image'][0])
    Y.append(train_data.__getitem__(i)['traveltime'])
X = np.asarray(X)
X = torch.from_numpy(X)
X = np.array(X.reshape(2924,675))   
Y = np.asarray(Y)
X_2d = bh_sne(X)
plt.scatter(X_2d[:,0],X_2d[:,1],c=Y)
plt.colorbar()