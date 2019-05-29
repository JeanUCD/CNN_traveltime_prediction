# -*- coding: utf-8 -*-
"""
Created on Mon May 20 22:02:02 2019

@author: Jean
"""

import pandas as pd
import os
import numpy as np
import glob

pwd = os.path.abspath('.') 
filepath = pwd + '/Data/Bluetoothindividual .xlsx'
df = pd.read_excel(filepath) 

def GetTravelTime_by5min(df,date_index):
    '''
    Params
    ------
        df: pandas dataframe of the traveltime data
        date_index: 0 for 10/15/2018, 1 for 10/16/2018...
    '''
    date_list = df.date.unique() # classify data by date
    segment_list = df.segment.unique()[0:3] # we just select the 3 segments esatbound
    df = df[df.date == date_list[date_index]] # choose the data of date_index
    TT = [] # list for travel time
    
    # The original data is arranged by timestamp, sample it every 5 minutes
    for i in range(len(segment_list)):
        # fill NA with the value before it 
        TT_segment = df[df.segment==segment_list[i]].traveltime.fillna(method='pad').as_matrix().reshape(-1,1).tolist()
        Time_segment = df[df.segment==segment_list[i]].time.fillna(method='pad').as_matrix().reshape(-1,1).tolist()
        nrow = len(Time_segment)
        TT_by5min = []
        for j in range(int(24*60/5)):
            TT_by5min.append(0)
            TT_buff = []
            time_mark_by5min_start = (j)*5/60/24
            time_mark_by5min_end = (j+1)*5/60/24
            for k in range(nrow-1):
                if (Time_segment[k][0] >= time_mark_by5min_start) and (Time_segment[k][0] <= time_mark_by5min_end):
                    TT_buff.append(TT_segment[k][0])                    
            # delete outliers
            TT_buff = np.array(TT_buff)
            TT_mean = np.mean(TT_buff,axis = 0)
            sd = np.std(TT_buff, axis = 0)
            TT_buff = [x for x in TT_buff if (x > TT_mean - 1 * sd)]
            TT_buff = [x for x in TT_buff if (x < TT_mean + 1 * sd)]
            TT_by5min[j] = np.mean(TT_buff)
        TT.append(TT_by5min)
    return TT
    
def SaveTravelTime_withImgName(df,date_index,datatype):
    '''
    Params
    ------
        df: pandas dataframe of the traveltime data
        date_index: 0 for 10/15/2018, 1 for 10/16/2018...
        datatype: flow, speed, occupancy
    '''
    TravelTime = GetTravelTime_by5min(df,date_index) # type: list
    # Aggregate the travel time of the 3 segments to AggTT
    AggTT = np.sum([TravelTime[0],TravelTime[1],TravelTime[2]],axis=0)# type: numpy ndarray
    # AggTT = Travel Time @ [1,2,3...] * 5 min    
    # for the first image stores the information of [0,14]* 5 min, AggTT starts from 15 * 5min
    AggTT = AggTT[14:len(AggTT)-1]
    AggTT = pd.DataFrame((AggTT))
    AggTT.columns = ['TravelTime']
    # fill 0 with mean
    AggTT.TravelTime[AggTT.TravelTime == 0] = AggTT.TravelTime.mean()
    # convert regression to classification
    AggTT.TravelTime[AggTT.TravelTime <= 700] = (AggTT.TravelTime - 420)//30
    AggTT.TravelTime[AggTT.TravelTime > 700] = 9
  
    # link the image path to corresponding travel time to be predicted    
    Image_name = [] 
    for i in range(AggTT.shape[0]):
        Image_name.append("{}_{}_{}.jpg".format(datatype,1015+date_index,i))
    AggTT['Img_name'] = pd.Series(Image_name)
    AggTT = AggTT.dropna(axis = 0, how = "any") # delete data with NAs
    AggTT.to_csv(pwd+"/TravelTime/{}_{}.csv".format(datatype, date_index))

# Save the csvs
for i in range(7):
    SaveTravelTime_withImgName(df,i,'flow')
    SaveTravelTime_withImgName(df,i,'speed')
    SaveTravelTime_withImgName(df,i,'occupancy')
    SaveTravelTime_withImgName(df,i,'observation')

# Get lists of csvs containing same types of data
all_flow_filenames = [i for i in glob.glob(pwd+'/TravelTime/flow_*.csv')]
all_speed_filenames = [i for i in glob.glob(pwd+'/TravelTime/speed_*.csv')]
all_occupancy_filenames = [i for i in glob.glob(pwd+'/TravelTime/occupancy_*.csv')]
all_observation_filenames = [i for i in glob.glob(pwd+'/TravelTime/observation_*.csv')]

# Combine all files in the list
combined_flow_csv = pd.concat([pd.read_csv(f) for f in all_flow_filenames ])
combined_speed_csv = pd.concat([pd.read_csv(f) for f in all_speed_filenames ])
combined_occupancy_csv = pd.concat([pd.read_csv(f) for f in all_occupancy_filenames ])
combined_observation_csv = pd.concat([pd.read_csv(f) for f in all_observation_filenames ])

# Export to csv
combined_flow_csv.to_csv( pwd+"/TravelTime/combined_flow.csv", index=False, encoding='utf-8-sig')
combined_speed_csv.to_csv( pwd+"/TravelTime/combined_speed.csv", index=False, encoding='utf-8-sig')
combined_occupancy_csv.to_csv( pwd+"/TravelTime/combined_occupancy.csv", index=False, encoding='utf-8-sig')
combined_observation_csv.to_csv( pwd+"/TravelTime/combined_observation.csv", index=False, encoding='utf-8-sig')
