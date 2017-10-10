#!C:\Python27\python
#-*- coding: UTF-8 -*-
import xlrd                    #导入xlrd模块 
import numpy as np
import matplotlib.pyplot as plt
 
  
#打开指定文件路径的excel文件  
  
xlsfile = r'E:\study\python\financial\icbc_py.xlsx'   
data = xlrd.open_workbook(xlsfile)     #获得excel的book对象  

table=data.sheets()[0]
nrows=table.nrows
ncols=table.ncols
r1 = table.row_values(3) #某一行数据
c1 = table.col_values(5)
list =[]
print c1[2:7]
#print nrows,ncols
#for rownum in range(2,nrows):
#    print table.row_values(0)

    


