#-*- coding: UTF-8 -*- 
import pandas as pd
xls_file=pd.ExcelFile(r'E:\study\python\financial\icbc_py.xlsx')
data=xls_file.parse('file')
#print data
closing_price=pd.DataFrame(data,columns=['closing_price'])
#element=pd.DataFrame（data,columns=['closing_price'],index=[1]）
#print closing_price.iloc[0]
for
