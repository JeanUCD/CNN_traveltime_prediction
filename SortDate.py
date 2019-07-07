# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 19:00:34 2019

@author: jean
"""

import pandas as pd
import os
import sys

sys.path.append("..")
pwd = os.path.abspath('.') 

normtype = '5min'
pth = pwd+'/TravelTime/{}'.format(normtype)

def AddOrder(normtype, settype, datatype):
    pth = pwd+'/TravelTime/{}'.format(normtype)
    df = pd.read_csv(pth+'/{}/combined_{}.csv'.format(settype,datatype))  
    df['seq'] = 0
    for row in range(df.shape[0]):
        df['seq'][row]  = ''.join([x for x in df.Img_name[row] if x.isdigit()])
    df.to_csv(pth+'/{}/combined_{}.csv'.format(settype,datatype))
    
AddOrder('5min','train_data','flow')
AddOrder('5min','train_data','occupancy')
AddOrder('5min','train_data','speed')
AddOrder('5min','train_data','observation')

AddOrder('no_normal_5min','train_data','flow')
AddOrder('no_normal_5min','train_data','occupancy')
AddOrder('no_normal_5min','train_data','speed')
AddOrder('no_normal_5min','train_data','observation')

AddOrder('uniform_5min','train_data','flow')
AddOrder('uniform_5min','train_data','occupancy')
AddOrder('uniform_5min','train_data','speed')
AddOrder('uniform_5min','train_data','observation')

AddOrder('5min','test_data','flow')
AddOrder('5min','test_data','occupancy')
AddOrder('5min','test_data','speed')
AddOrder('5min','test_data','observation')

AddOrder('no_normal_5min','test_data','flow')
AddOrder('no_normal_5min','test_data','occupancy')
AddOrder('no_normal_5min','test_data','speed')
AddOrder('no_normal_5min','test_data','observation')

AddOrder('uniform_5min','test_data','flow')
AddOrder('uniform_5min','test_data','occupancy')
AddOrder('uniform_5min','test_data','speed')
AddOrder('uniform_5min','test_data','observation')

