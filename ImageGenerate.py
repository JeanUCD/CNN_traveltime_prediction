# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:55:44 2019

@author: Jean
"""
"""
Generate images from PeMS data
Convert flow rate/speed/occupancy/observation rate data to images as:

       t     t+5min    t+10min   ....   ....  t+5min*window_len     
VDS15  f/s/o   f/s/o     f/s/o                  f/s/o
VDS14  f/s/o   f/s/o     f/s/o                  f/s/o
VDS13  f/s/o   f/s/o     f/s/o                  f/s/o
....  .....   .....     .....
....
....
VDS1   f/s/o   f/s/o     f/s/0                  f/s/o


"""
import pandas as pd
import os
import numpy as np
import scipy.misc
import glob

# to get the range of flow,occupancy and speed for image normalization purpose
pwd = os.path.abspath('.') 
all_flow_filenames = [i for i in glob.glob(pwd+'/Data/*_flow.xlsx')]
all_speed_filenames = [i for i in glob.glob(pwd+'/Data/*_speed.xlsx')]
all_occupancy_filenames = [i for i in glob.glob(pwd+'/Data/*_occupancy.xlsx')]
all_flow_csv = pd.concat([pd.read_excel(f) for f in all_flow_filenames ])
all_speed_csv = pd.concat([pd.read_excel(f) for f in all_speed_filenames ])
all_occupancy_csv = pd.concat([pd.read_excel(f) for f in all_occupancy_filenames ])
all_flow_csv.to_csv( pwd+"/Data/all_flow.csv", index=False, encoding='utf-8-sig') #0-540
all_speed_csv.to_csv( pwd+"/Data/all_speed.csv", index=False, encoding='utf-8-sig')#4.3-78.4
all_occupancy_csv.to_csv( pwd+"/Data/all_occupancy.csv", index=False, encoding='utf-8-sig')#0-0.65

def GetFlowMatrix(df):
    '''
    Get the data matrix of flow rates of the 15 VDSs
            t    t+5min  t+10min   ....
    VDS15    
    VDS14
    ....
    VDS1  
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
    VDS15    
    VDS14
    ....
    VDS1  
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
    VDS15    
    VDS14
    ....
    VDS1  
    '''
    
    nrow = df.shape[0]
    VDS_list = df.VDS.unique()
    Frame = np.zeros(shape = (1,int(nrow/len(VDS_list))))
    for i in range(len(VDS_list)):
        VDS_data = df[df.VDS==VDS_list[i]].AggSpeed.fillna(method='pad').as_matrix().reshape(1,-1)
        Frame = np.concatenate((Frame,VDS_data), axis=0)
    Frame = np.delete(Frame,(0),axis=0)
    return Frame

def GetObRateMatrix(df):
    '''
    Get the data matrix of observation rate of the 15 VDSs
            t    t+5min  t+10min   ....
    VDS15    
    VDS14
    ....
    VDS1  
    '''
    
    nrow = df.shape[0]
    VDS_list = df.VDS.unique()
    Frame = np.zeros(shape = (1,int(nrow/len(VDS_list))))
    for i in range(len(VDS_list)):
        VDS_data = df[df.VDS==VDS_list[i]]['% Observed'].fillna(method='pad').as_matrix().reshape(1,-1)
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
        date: 0 for 10/15, 1 for 10/16 .....
    '''
    mat = GetDataFun(df)
    mat_len = mat.shape[1]
    pwd = os.path.abspath('.')
    for i in range(mat_len - window_len):
        mat_window = mat[:,i:i+window_len]
        out_path = pwd+'/Pics/{}/{}_{}_{}.jpg'.format(datatype,datatype,date,i)
        if datatype == 'flow':
            scipy.misc.toimage(mat_window, cmin = 0.0, cmax = 540).save(out_path)
        elif datatype == 'speed':
            scipy.misc.toimage(mat_window, cmin = 4.3, cmax = 78.4).save(out_path)
        elif datatype == 'occupancy':
            scipy.misc.toimage(mat_window, cmin = 0.0, cmax = 0.65).save(out_path)
        else:
            scipy.misc.toimage(mat_window, cmin = 0.0, cmax = 100).save(out_path)

# Save the data of total 31 days to images
for i in range(31):
    if i > 16:
        date = 1101+i-17
    else:
        date = 1015 + i
    filepath_flow = pwd + '/Data/{}_flow.xlsx'.format(date)
    filepath_speed = pwd + '/Data/{}_speed.xlsx'.format(date)
    filepath_occupancy = pwd + '/Data/{}_occupancy.xlsx'.format(date)       
    SaveImage(pd.read_excel(filepath_flow),15,GetFlowMatrix,'flow','{}'.format(date))
    SaveImage(pd.read_excel(filepath_speed),15,GetSpeedMatrix,'speed','{}'.format(date))
    SaveImage(pd.read_excel(filepath_occupancy),15,GetOccupancyMatrix,'occupancy','{}'.format(date))
    SaveImage(pd.read_excel(filepath_speed),15,GetObRateMatrix,'observation','{}'.format(date))
