# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 09:25:44 2019

@author: jean
"""

import pandas as pd
import os
import sys
from skimage import io, transform
import numpy as np

sys.path.append("..")
pwd = os.path.abspath('.') 

normtype = '5min'
settype = 'train_data'
datatype = 'flow'
Pic_pth = pwd+'/Pics/observation'
pth = pwd+'/TravelTime/{}'.format(normtype)
df = pd.read_csv(pth+'/{}/combined_{}.csv'.format(settype,datatype))  
df['ob']=0.0
df['ObImg']=df.iloc[:,3]
for i in range(df.shape[0]):
    i = 10
    df['ObImg'][i] = df['ObImg'][i].replace('flow','observation')
    and_pth = df['ObImg'][i]
    img = io.imread(Pic_pth + '/'+and_pth)
    df['ob'][i] = np.mean(img)/255