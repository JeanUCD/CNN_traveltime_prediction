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

    def __init__(self, csv_file1, csv_file2, csv_file3, root_dir1, root_dir2, root_dir3, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.traveltime_frame1 = pd.read_csv(csv_file1)
        self.root_dir1 = root_dir1
        self.traveltime_frame2 = pd.read_csv(csv_file2)
        self.root_dir2 = root_dir2
        self.traveltime_frame3 = pd.read_csv(csv_file3)
        self.root_dir3 = root_dir3
        self.transform = transform

    def __len__(self):
        return len(self.traveltime_frame1)

    def __getitem__(self, idx):
        img_name1 = os.path.join(self.root_dir1,
                                self.traveltime_frame1.iloc[idx, 3])
        img_name2 = os.path.join(self.root_dir2,
                                self.traveltime_frame2.iloc[idx, 3])
        img_name3 = os.path.join(self.root_dir3,
                                self.traveltime_frame3.iloc[idx, 3])
        # Normalize the pixel values to [0,1]
        image1 = io.imread(img_name1)
        image2 = io.imread(img_name2)
        image3 = io.imread(img_name3)
        traveltime = self.traveltime_frame1.iloc[idx, 2]
        image = np.array([[image1,image2,image3]])
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
        image = image.reshape(3,15,15)
        traveltime = torch.from_numpy(np.array(traveltime)).float()
        return (image, traveltime)


