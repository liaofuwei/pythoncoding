#!C:\Python27\python
#-*- coding: UTF-8 -*-
import csv                   #导入csv模块 
import numpy as np
import matplotlib.pyplot as plt
 
  

with open('E:\study\python\EBR_data\EBR-from2008.csv','rb')as csvfile:
    reader=csv.reader(csvfile,delimiter=' ')
    for colmn in reader:
        print colmn

