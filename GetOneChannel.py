# -*- coding: utf-8 -*-
"""
Created on Fri May 24 20:41:08 2019

@author: jean
"""
from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

pwd = os.path.abspath('.') 


class TTdataset(Dataset):
    """Travel Time dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.traveltime_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.traveltime_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.traveltime_frame.iloc[idx, 3])
        # Normalize the pixel values to [0,1]
        image = io.imread(img_name)/255
        traveltime = self.traveltime_frame.iloc[idx, 2]
        sample = {'image': image, 'traveltime': traveltime}

        if self.transform:
            sample = self.transform(sample)

        return sample
        
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, traveltime = sample['image'], sample['traveltime']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        #image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image).float()
        image = image.reshape(1,15,15)
        traveltime = torch.from_numpy(np.array(traveltime)).float()
        return (image, traveltime)


