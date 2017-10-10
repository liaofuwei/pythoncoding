#!C:\Python27\python
#-*- coding: UTF-8 -*-
import xlrd                    #导入xlrd模块
import xlwt
import numpy as np
import matplotlib.pyplot as plt
 
  
#打开指定文件路径的excel文件  
  
xlsfile = r'E:\pythoncoding\treasury\treasury field.xls'   
data = xlrd.open_workbook(xlsfile)     #获得excel的book对象  

table=data.sheets()[0]
nrows=table.nrows
ncols=table.ncols
wb=xlwt.Workbook()
ws=wb.add_sheet('0')
#r1 = table.row_values(3) #某一行数据

cA = table.col_values(0) #5-year date
cC = table.col_values(2) #5-Year field
cH = table.col_values(7) #5-Year date
cK = table.col_values(10) #5-Year field

f

c0.reverse()
c1.reverse()
for i in range(0,404):
    ws.write(i,3,c0[i])
    ws.write(i,4,c1[i])
wb.save(r'C:\Users\Administrator\Desktop\treasury\5yearssss.xls')
