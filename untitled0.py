# -*- coding: utf-8 -*-
"""
Created on Sat Jun 22 15:26:42 2019

@author: jean
"""
'''
import googlemaps
from datetime import datetime

gmaps = googlemaps.Client(key='AIzaSyCAJFWuAfotdwAd2Keo2BDM09SoNo3efko')

now = datetime.now()

directions_result = gmaps.directions("37.8248,122.3139",
                                     "4010 Lake Rd, West Sacramento, CA 95691",
                                     mode = "driving",
                                     departure_time = now)
print(directions_result[0]['legs'][0]['duration']['text'])      
'''
                               