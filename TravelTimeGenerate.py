# -*- coding: utf-8 -*-
"""
Created on Mon May 20 22:02:02 2019

@author: Jean
"""

import pandas as pd
import os
import numpy as np


pwd = os.path.abspath('.') 
filepath = pwd + '/Data/Bluetoothindividual .xlsx'
df = pd.read_excel(filepath) 

def GetTravelTime_by5min(df,date_index):
    date_list = df.date.unique()
    segment_list = df.segment.unique()[0:3]
    df = df[df.date == date_list[date_index]]
    TT = []
    for i in range(len(segment_list)):
        TT_segment = df[df.segment==segment_list[i]].traveltime.fillna(method='pad').as_matrix().reshape(-1,1).tolist()
        Time_segment = df[df.segment==segment_list[i]].time.fillna(method='pad').as_matrix().reshape(-1,1).tolist()
        nrow = len(Time_segment)
        TT_by5min = []
        for j in range(int(24*60/5)):
            TT_by5min.append(0)
            time_mark_by5min = (j+1)*5/60/24
            for k in range(nrow-1):
                if Time_segment[k][0] < time_mark_by5min and Time_segment[k+1][0] > time_mark_by5min:
                    TT_by5min[j] = TT_segment[k][0]
        TT.append(TT_by5min)
    return TT
                    
TravelTime = GetTravelTime_by5min(df,0)