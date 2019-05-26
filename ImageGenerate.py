# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:55:44 2019

@author: Jean
"""
"""
Generate images from PeMS data
Convert flow rate/speed/occupancy data to images as:

       t     t+5min    t+10min   ....   ....  t+5min*window_len     
VDS1  f/s/o   f/s/o     f/s/o                  f/s/o
VDS2  f/s/o   f/s/o     f/s/o                  f/s/o
VDS3  f/s/o   f/s/o     f/s/o                  f/s/o
....
....
....
VDS15 f/s/o   f/s/o     f/s/0                  f/s/o


"""
import pandas as pd
import os
import numpy as np
import scipy.misc


def GetFlowMatrix(df):
    '''
    Get the data matrix of flow rates of the 15 VDSs
            t    t+5min  t+10min   ....
    VDS1    
    VDS2
    ....
    VDS15  
    '''
    nrow = df.shape[0]
    VDS_list = df.VDS.unique()
    Frame = np.zeros(shape = (1,int(nrow/len(VDS_list))))
    for i in range(len(VDS_list)):
        VDS_data = df[df.VDS==VDS_list[i]].AggFlow.fillna(method='pad').as_matrix().reshape(1,-1)
        Frame = np.concatenate((Frame,VDS_data), axis=0)
    Frame = np.delete(Frame,(0),axis=0)
    return Frame

def GetOccupancyMatrix(df):
    '''
    Get the data matrix of occupancy of the 15 VDSs
            t    t+5min  t+10min   ....
    VDS1    
    VDS2
    ....
    VDS15  
    '''
    nrow = df.shape[0]
    VDS_list = df.VDS.unique()
    Frame = np.zeros(shape = (1,int(nrow/len(VDS_list))))
    for i in range(len(VDS_list)):
        VDS_data = df[df.VDS==VDS_list[i]].AggOccupancy.fillna(method='pad').as_matrix().reshape(1,-1)
        Frame = np.concatenate((Frame,VDS_data), axis=0)
    Frame = np.delete(Frame,(0),axis=0)
    return Frame

def GetSpeedMatrix(df):
    '''
    Get the data matrix of speed of the 15 VDSs
            t    t+5min  t+10min   ....
    VDS1    
    VDS2
    ....
    VDS15  
    '''
    
    nrow = df.shape[0]
    VDS_list = df.VDS.unique()
    Frame = np.zeros(shape = (1,int(nrow/len(VDS_list))))
    for i in range(len(VDS_list)):
        VDS_data = df[df.VDS==VDS_list[i]].AggSpeed.fillna(method='pad').as_matrix().reshape(1,-1)
        Frame = np.concatenate((Frame,VDS_data), axis=0)
    Frame = np.delete(Frame,(0),axis=0)
    return Frame


def SaveImage(df, window_len, GetDataFun, datatype, date):
    '''
    Save images by datatype(flow/occupancy/speed) and date(10/14,10/15...)
    
    Params
    ------
        df: the original data set
        window_len: the length of the images(time window)
        GetDataFun: GetFlowMatrix, GetOccupancyMatrix, GetSpeedMatrix
        datatype: flow, occupancy, speed
        date: 0 for 10/14, 1 for 10/15 .....
    '''
    mat = GetDataFun(df)
    mat_len = mat.shape[1]
    pwd = os.path.abspath('.') 
    for i in range(mat_len - window_len):
        mat_window = mat[:,i:i+window_len]
        out_path = pwd+'/Pics/{}/{}_{}_{}.jpg'.format(datatype,datatype,date,i)
        scipy.misc.imsave(out_path, mat_window)

pwd = os.path.abspath('.')
# Save the data of total 8 days to images
for i in range(8): 
    filepath_flow = pwd + '/Data/{}_flow.xlsx'.format(1014+i)
    filepath_speed = pwd + '/Data/{}_speed.xlsx'.format(1014+i)
    filepath_occupancy = pwd + '/Data/{}_occupancy.xlsx'.format(1014+i)       
    SaveImage(pd.read_excel(filepath_flow),15,GetFlowMatrix,'flow','{}'.format(1014+i))
    SaveImage(pd.read_excel(filepath_speed),15,GetSpeedMatrix,'speed','{}'.format(1014+i))
    SaveImage(pd.read_excel(filepath_occupancy),15,GetOccupancyMatrix,'occupancy','{}'.format(1014+i))
