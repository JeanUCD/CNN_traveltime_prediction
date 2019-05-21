# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:55:44 2019

@author: Jean
"""

import pandas as pd
import os
import numpy as np
import scipy.misc
import cv2
from PIL import Image

def GetFlowMatrix(df):
    nrow = df.shape[0]
    VDS_list = df.VDS.unique()
    Frame = np.zeros(shape = (1,int(nrow/len(VDS_list))))
    for i in range(len(VDS_list)):
        VDS_data = df[df.VDS==VDS_list[i]].AggFlow.fillna(method='pad').as_matrix().reshape(1,-1)
        Frame = np.concatenate((Frame,VDS_data), axis=0)
    Frame = np.delete(Frame,(0),axis=0)
    return Frame

def GetOccupancyMatrix(df):
    nrow = df.shape[0]
    VDS_list = df.VDS.unique()
    Frame = np.zeros(shape = (1,int(nrow/len(VDS_list))))
    for i in range(len(VDS_list)):
        VDS_data = df[df.VDS==VDS_list[i]].AggOccupancy.fillna(method='pad').as_matrix().reshape(1,-1)
        Frame = np.concatenate((Frame,VDS_data), axis=0)
    Frame = np.delete(Frame,(0),axis=0)
    return Frame

def GetSpeedMatrix(df):
    nrow = df.shape[0]
    VDS_list = df.VDS.unique()
    Frame = np.zeros(shape = (1,int(nrow/len(VDS_list))))
    for i in range(len(VDS_list)):
        VDS_data = df[df.VDS==VDS_list[i]].AggSpeed.fillna(method='pad').as_matrix().reshape(1,-1)
        Frame = np.concatenate((Frame,VDS_data), axis=0)
    Frame = np.delete(Frame,(0),axis=0)
    return Frame


def SaveImage(df, window_len, GetDataFun, datatype, date):
    mat = GetDataFun(df)
    mat_len = mat.shape[1]
    pwd = os.path.abspath('.') 
    for i in range(mat_len - window_len):
        mat_window = mat[:,i:i+window_len]
        out_path = pwd+'/Pics/{}/{}_{}_{}.jpg'.format(datatype,datatype,date,i)
        scipy.misc.imsave(out_path, mat_window)

pwd = os.path.abspath('.') 
filepath = pwd + '/Data/1014_speed.xlsx'
df = pd.read_excel(filepath)       
SaveImage(df,15,GetSpeedMatrix,'speed','1014')


