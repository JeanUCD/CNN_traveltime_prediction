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
'''
filepath = pwd + '/Data/Bluetoothindividual .xlsx'
df = pd.read_excel(filepath) 

# Get rid of outliers
df.traveltime = df.traveltime.fillna(method='pad')
pd.options.mode.chained_assignment = None
for i in range(df.shape[0]):
    if i > 0:
        temp = 0
        if df.segment[i] == df.segment[i-1]:
            if (df.traveltime[i] > 2*df.traveltime[i-1]) or (df.traveltime[i] < 0.5*df.traveltime[i-1]) :
                temp = df.traveltime[i-1]
                df.traveltime[i] = temp
df.to_csv(pwd+"/Data/TravelTime_filtered.csv", index=False, encoding='utf-8-sig')
'''

df = pd.read_csv(pwd+"/Data/TravelTime_filtered.csv")

def GetTravelTime_by5min(df,date_index):
    '''
    Params
    ------
        df: pandas dataframe of the traveltime data
        date_index: 0 for 10/14/2018, 1 for 10/15/2018...
    '''
    date_list = df.date.unique() # classify data by date
    segment_list = df.segment.unique()[0:3] # we just select the 3 segments esatbound
    df = df[df.date == date_list[date_index]] # choose the data of date_index
    TT_mean = [] # list for travel time mean
    TT_std = [] # list for travel time std
    
    # The original data is arranged by timestamp, sample it every 5 minutes
    for i in range(len(segment_list)):
        # fill NA with the value before it 
        TT_segment = df[df.segment==segment_list[i]].traveltime.fillna(method='pad').as_matrix().reshape(-1,1).tolist()
        #Time_segment = df[df.segment==segment_list[i]].time.fillna(method='pad').as_matrix().reshape(-1,1).tolist()
        Time_segment = df[df.segment==segment_list[i]].time.as_matrix().reshape(-1,1).tolist()
        nrow = len(Time_segment)
        avg_5min = []
        std_5min = []
        for j in range(int(24*60/5)):
            avg_5min.append(0)
            std_5min.append(0)
            TT_buff = []
            time_mark_by5min_start = (j)*5/60/24
            time_mark_by5min_end = (j+1)*5/60/24
            for k in range(nrow-1):
                if (Time_segment[k][0] >= time_mark_by5min_start) and (Time_segment[k][0] <= time_mark_by5min_end):
                    TT_buff.append(TT_segment[k][0])                    
            # delete outliers
            TT_buff = np.array(TT_buff)
            avg_5min[j] = np.mean(TT_buff,axis = 0)
            std_5min[j] = np.std(TT_buff, axis = 0)
            #TT_buff = [x for x in TT_buff if (x > TT_mean - 1 * sd)]
            #TT_buff = [x for x in TT_buff if (x < TT_mean + 1 * sd)]
            #TT_by5min[j] = np.mean(TT_buff,axis = 0)
        TT_mean.append(avg_5min)
        TT_std.append(std_5min)
    return TT_mean,TT_std
    
def SaveTravelTime_withImgName(df,date_index,datatype,pred_interval):
    '''
    Params
    ------
        df: pandas dataframe of the traveltime data
        date_index: 0 for 10/14/2018, 1 for 10/15/2018...
        datatype: flow, speed, occupancy
        pred_interval: '5min', '30min'
    '''
    TravelTime,_ = GetTravelTime_by5min(df,date_index) # type: list
    # Aggregate the travel time of the 3 segments to AggTT
    AggTT = np.sum([TravelTime[0],TravelTime[1],TravelTime[2]],axis=0)# type: numpy ndarray
    # AggTT = Travel Time @ [1,2,3...] * 5 min
    if pred_interval == '5min':    
    # for the first image stores the information of [0,14]* 5 min, AggTT starts from 15 * 5min
        AggTT = AggTT[14:len(AggTT)-1]
    else:
        TravelTime_nextday,_ = GetTravelTime_by5min(df,date_index+1)
        AggTT_nextday = np.sum([TravelTime_nextday[0],TravelTime_nextday[1],TravelTime_nextday[2]],axis=0)
        AggTT = AggTT[20:len(AggTT)-1]+AggTT_nextday[0:5]
    AggTT = pd.DataFrame((AggTT))
    AggTT.columns = ['TravelTime']
    # fill 0 with mean
    AggTT.TravelTime[AggTT.TravelTime == 0] = AggTT.TravelTime.mean()
    '''
    # convert regression to classification
    AggTT.TravelTime[AggTT.TravelTime <= 749] = (AggTT.TravelTime - 450)//30
    AggTT.TravelTime[AggTT.TravelTime > 749] = 9
    list1 = [0 if i < 0 else i for i in AggTT.TravelTime]
    AggTT.TravelTime = list1
    '''
    # normalize to 0-1
    # uniform normal
    '''
    for i in range(AggTT.shape[0]):
        AggTT.TravelTime[i] = min(1,max(0,(AggTT.TravelTime[i]-420)/(1500-420)))
    '''
    # Z-Score normal
    # mean = 676.44
    # std  = 408.85
    for i in range(AggTT.shape[0]):
        AggTT.TravelTime[i] = (AggTT.TravelTime[i]-676.44)/408.85
    
    
    # link the image path to corresponding travel time to be predicted    
    Image_name = [] 
    if date_index > 17:
        date = 1101+date_index-18
    else:
        date = 1014 + date_index    
    for i in range(AggTT.shape[0]):
        Image_name.append("{}_{}_{}.jpg".format(datatype,date,i))
    AggTT['Img_name'] = pd.Series(Image_name)
    AggTT = AggTT.dropna(axis = 0, how = "any") # delete data with NAs
    AggTT.to_csv(pwd+"/TravelTime/{}pred_{}_{}.csv".format(pred_interval,datatype, date_index))

# Save the csvs
pred_interval = '5min'

for i in range(33):
    SaveTravelTime_withImgName(df,i,'flow',pred_interval)
    SaveTravelTime_withImgName(df,i,'speed',pred_interval)
    SaveTravelTime_withImgName(df,i,'occupancy',pred_interval)
    SaveTravelTime_withImgName(df,i,'observation',pred_interval)

# Get lists of csvs containing same types of data
set_type = 'test_data'
pred_interval = '5min'
all_flow_filenames = [i for i in glob.glob(pwd+'/TravelTime/{}/{}/{}pred_flow_*.csv'.format(pred_interval,set_type,pred_interval))]
all_speed_filenames = [i for i in glob.glob(pwd+'/TravelTime/{}/{}/{}pred_speed_*.csv'.format(pred_interval,set_type,pred_interval))]
all_occupancy_filenames = [i for i in glob.glob(pwd+'/TravelTime/{}/{}/{}pred_occupancy_*.csv'.format(pred_interval,set_type,pred_interval))]
all_observation_filenames = [i for i in glob.glob(pwd+'/TravelTime/{}/{}/{}pred_observation_*.csv'.format(pred_interval,set_type,pred_interval))]

# Combine all files in the list
combined_flow_csv = pd.concat([pd.read_csv(f) for f in all_flow_filenames ])
combined_speed_csv = pd.concat([pd.read_csv(f) for f in all_speed_filenames ])
combined_occupancy_csv = pd.concat([pd.read_csv(f) for f in all_occupancy_filenames ])
combined_observation_csv = pd.concat([pd.read_csv(f) for f in all_observation_filenames ])

# Export to csv
combined_flow_csv.to_csv( pwd+"/TravelTime/{}/{}/combined_flow.csv".format(pred_interval,set_type), index=False, encoding='utf-8-sig')
combined_speed_csv.to_csv( pwd+"/TravelTime/{}/{}/combined_speed.csv".format(pred_interval,set_type), index=False, encoding='utf-8-sig')
combined_occupancy_csv.to_csv( pwd+"/TravelTime/{}/{}/combined_occupancy.csv".format(pred_interval,set_type), index=False, encoding='utf-8-sig')
combined_observation_csv.to_csv( pwd+"/TravelTime/{}/{}/combined_observation.csv".format(pred_interval,set_type), index=False, encoding='utf-8-sig')

