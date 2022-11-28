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

from numpy import dtype

#load data
def Read_File(filename):
    file = np.loadtxt(filename, dtype = np.float)
    return file
train_data = Read_File('D_RSSI.txt')
Tr_coor = Read_File('Tr_coor.txt')
Rssi_Data = Read_File('Rssi_data.txt')

Rssi_Data_verify = Read_File('Rssi_data_verify.txt')
Veri_coor = Read_File('Veri_coor.txt')

#set KNN

def Get_Dist(x):
    if x>59:
        if x<70:
            k = 5
        else: k = 2
    else: k = 2
    distances = []
    for i in train_data:
        d = sqrt((i[1] - x)**2)
        distances.append(d)
    nearest = np.argsort(distances)
    topk = [[train_data[i,0]] for i in nearest[:k]]
    mean_topk = np.mean(topk)
    return mean_topk


k_list = []
for i in range(0,10):
    k_list.append(5)
print(k_list)

def func(n):
    if n == 0:
        return k_list
    else:
        return func(n-1)[n]+1
print(func(10))
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
#train_data_1 = Read_File('test.txt')
Rssi_Data_verify = Read_File('Rssi_data_verify.txt')
Veri_coor = Read_File('Veri_coor.txt')
train_data_1 = []

#Fixed KNN algorithm(k=5 or k=2)
def Get_Dist(x):
    if x>59:
        if x<73:
            k = 5
        else: k = 2
    else: k = 2
    distances = []
    for i in train_data:
        d = sqrt((i[1] - x)**2)
        distances.append(d)
    nearest = np.argsort(distances)
    topk = [train_data[i,0] for i in nearest[:k]]
    mean_topk = np.mean(topk)
    train_data_1.append(mean_topk)
    #print(mean_topk)
    return mean_topk

for x in range(45,101):
    Get_Dist(x)
#print(train_data_1)


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



#calculate the the targets' coordinate and error by triposition
def Loacate(x_data):
    sum1 = 0
    dist = [0,0,0,0]
    for a in range(0,13):
        line_list = [Rssi_Data[0,a+1],Rssi_Data[1,a+1],Rssi_Data[2,a+1],Rssi_Data[3,a+1]]
        idx = sorted(enumerate(line_list),key = lambda x: x[1])
        b = int(idx[0][1])
        c = int(idx[1][1])
        d = int(idx[2][1])  
        dist[idx[0][0]] = x_data[b-45]
        dist[idx[1][0]] = x_data[c-45]
        dist[idx[2][0]] = x_data[d-45]
        #estimate obstacle--AP distance
        if line_list[1]>81:
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
        #print(sqrt((locx - Veri_coor[a,0])**2+(locy-Veri_coor[a,1])**2))
        sum1 = sum1 + (locx - Tr_coor[a,0])**2+(locy-Tr_coor[a,1])**2
    print('Average=%.7f' % sqrt(sum1/13))
    return sum1

x_data = train_data_1
print(x_data)
learnrate = 0.1
w = {}
for i in range(0,101):
      w[i] = 1


x_data[15] = 33.63094
x_data[16] = 10
x_data[17] = 11
y_true = Loacate(x_data)





for i in range(63,73):
    iter = 1
    while iter <= 25:
        x_data[i-45] = x_data[i-45]+w[i]
        y_pred = Loacate(x_data)
        if y_pred>y_true:
            w[i] = -w[i]
            if w[i]<0:
                w[i] = w[i]+learnrate
            else: 
                w[i]-learnrate
                y_true = y_pred
            #b[i] = random.uniform(-0.1,0.1)
        print(w[i], x_data[i-45], iter )
        iter += 1

print(x_data)