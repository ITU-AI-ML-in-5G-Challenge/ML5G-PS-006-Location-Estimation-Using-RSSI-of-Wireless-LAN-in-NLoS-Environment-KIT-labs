from ctypes import sizeof
import enum
from tokenize import group
from turtle import distance
from numpy import *
import numpy as np
import random
import operator
import matplotlib.pyplot as plt
from math import sqrt
import statistics
import sympy
import tensorflow as tf
import pandas as pd 
import csv
import os

from numpy import dtype

#load data
def Read_File(filename):
    file = np.loadtxt(filename, dtype = np.float)
    return file
train_data = Read_File('D_RSSI.txt')
#train_data_1 = Read_File('test.txt')
Tr_coor = Read_File('Tr_coor.txt')
Rssi_Data = Read_File('Rssi_data.txt')
train_data_1 = Read_File('test.txt')
veri_results = Read_File('verification.txt')
Rssi_Data_verify = Read_File('Rssi_data_verify.txt')
Veri_coor = Read_File('Veri_coor.txt')



#set converted AP coordinate in meters
Coordinate_Data = [[35.945064, 0], 
                   [0, 2.073498],
                   [11.499303, 29.752712],
                   [43.514789, 24.825213]]
AP_coor = np.array(Coordinate_Data)

#triposition algorithm
def triposition(xa,ya,da,xb,yb,db,xc,yc,dc): 
    x,y = sympy.symbols('x y')
    f1 = 2*x*(xa-xc)+np.square(xc)-np.square(xa)+2*y*(ya-yc)+np.square(yc)-np.square(ya)-(np.square(dc)-np.square(da))
    f2 = 2*x*(xb-xc)+np.square(xc)-np.square(xb)+2*y*(yb-yc)+np.square(yc)-np.square(yb)-(np.square(dc)-np.square(db))
    result = sympy.solve([f1,f2],[x,y])
    locx,locy = result[x],result[y]
    return [locx,locy]


dist = [0,0,0,0]

#calculate the the targets' coordinate and error by triposition
def Loacate():
    sum1 = 0
    for a in range(0,13):
        line_list = [Rssi_Data_verify[0,a+1],Rssi_Data_verify[1,a+1],Rssi_Data_verify[2,a+1],Rssi_Data_verify[3,a+1]]
        idx = sorted(enumerate(line_list),key = lambda x: x[1])
        dist[idx[0][0]] = veri_results[int(idx[0][1])-45,1]
        dist[idx[1][0]] = veri_results[int(idx[1][1])-45,1]
        dist[idx[2][0]] = veri_results[int(idx[2][1])-45,1]

    #estimate obstacle--AP distance
        if line_list[1]>83:
            if line_list[3]<60:
                dist[1] = 45.3124
        if line_list[3]>78:
            if line_list[1]<65:
                dist[3] = 36.6082
        if line_list[2]>81:
            if line_list[0]<65:
                if line_list[1]<70:
                    dist[2] = 26.9626

            #print(dist[idx[0][0]], dist[idx[1][0]], dist[idx[2][0]])
            #print(line_list)
            #print(idx[0][1],idx[1][1],idx[2][1])
            #print(idx[0][0],idx[1][0],idx[2][0])
        idx_1 = idx[0][0]
        idx_2 = idx[1][0]
        idx_3 = idx[2][0]
        [locx,locy] = triposition(AP_coor[idx_1,0],AP_coor[idx_1,1],
                dist[idx[0][0]],AP_coor[idx_2,0],AP_coor[idx_2,1],dist[idx[1][0]],AP_coor[idx_3,0],AP_coor[idx_3,1],dist[idx[2][0]])
        #print(locx,locy,k)
        print(sqrt((locx - Veri_coor[a,0])**2+(locy-Veri_coor[a,1])**2))
        sum1 = sum1 + (locx - Veri_coor[a,0])**2+(locy-Veri_coor[a,1])**2
    print('Average=%.7f' % sqrt(sum1/13))

Loacate()



