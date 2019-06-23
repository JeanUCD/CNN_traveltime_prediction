# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:37:45 2019

@author: jean
"""

import cv2
import numpy as np
import glob

img_array = []
for filename in glob.glob('Pics/speed/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
    
out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'),15, size)

for i in range(len(img_array)):
    out.write(img_array[i])
out.release()     