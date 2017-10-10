__author__ = 'Administrator'
import quandl
import datetime
import pandas as pd
from bs4 import BeautifulSoup
import requests
import time
import urllib
import numpy as np



'''
从新浪行情获取comex 黄金实时数据
并以添加模式写入到csv中
'''
COMEXgold_Url ="http://hq.sinajs.cn/list=hf_GC"
COMEXgold_web_data = requests.get(COMEXgold_Url)
COMEXgold_soup = BeautifulSoup(COMEXgold_web_data.text,"lxml")
rawdata=COMEXgold_soup.get_text()
rawdataList=rawdata.split("\"")
data = rawdataList[1].split(",")
if len(data)==14:
    comex_gold={
        "real_price":data[0],
        "price_change":data[1],
        "buy_price":data[2],
        "sell_price":data[3],
        "high":data[4],
        "low":data[5],
        "time":data[6],
        "close_price":data[7],
        "open_price":data[8],
        "pisition":data[9],
        "date":data[12]
    }

#comexData=pd.Series(comex_gold,index=['date','time','real_price','price_change','buy_price','sell_price','high','low','close_price',\
#                                      'open_price','pisition'])

    comexData=pd.DataFrame(comex_gold,columns=['date','time','real_price','price_change','buy_price','sell_price','high','low','close_price',\
                                               'open_price','pisition'],index=[0])
    dt=str(datetime.datetime.now().date())
#comexData.to_csv("E:\\pythoncoding\\data\\gold\\COMEX_gold_"+dt+".csv")
#以添加模式加入到csv文件中
    comexData.to_csv("E:\\pythoncoding\\data\\gold\\COMEX_gold_"+dt+".csv",mode='a',header=False)  



'''
从新浪行情获取london 黄金实时数据
并以添加模式写入到csv中
'''
Londongold_Url ="http://hq.sinajs.cn/list=hf_XAU"
Londongold_web_data = requests.get(Londongold_Url)
Londongold_soup = BeautifulSoup(Londongold_web_data.text,"lxml")
rawdata=Londongold_soup.get_text()
rawdataList=rawdata.split("\"")
data = rawdataList[1].split(",")

if len(data)==14:
    London_gold={
        "real_price":data[0],
        "price_change":data[1],
        "buy_price":data[2],
        "sell_price":data[3],
        "high":data[4],
        "low":data[5],
        "time":data[6],
        "close_price":data[7],
        "open_price":data[8],
        "pisition":data[9],
        "date":data[12]
    }



    #comexData=pd.Series(comex_gold,index=['date','time','real_price','price_change','buy_price','sell_price','high','low','close_price',\
    #                                      'open_price','pisition'])

    LondonData=pd.DataFrame(London_gold,columns=['date','time','real_price','price_change','buy_price','sell_price','high','low','close_price',\
                                                'open_price','pisition'],index=[0])
    dt=str(datetime.datetime.now().date())
#LondonData.to_csv("E:\\pythoncoding\\data\\gold\\London_gold_"+dt+".csv")
#以添加模式加入到csv文件中
    LondonData.to_csv("E:\\pythoncoding\\data\\gold\\London_gold_"+dt+".csv",mode='a',header=False)                                           

'''
从新浪行情获取comex 白银实时数据
并以添加模式写入到csv中
'''
COMEXsilver_Url ="http://hq.sinajs.cn/list=hf_SI"
COMEXsilver_web_data = requests.get(COMEXsilver_Url)
COMEXsilver_soup = BeautifulSoup(COMEXsilver_web_data.text,"lxml")
rawdata=COMEXsilver_soup.get_text()
rawdataList=rawdata.split("\"")
data = rawdataList[1].split(",")
if len(data)==14:
    comex_silver={
        "real_price":data[0],
        "price_change":data[1],
        "buy_price":data[2],
        "sell_price":data[3],
        "high":data[4],
        "low":data[5],
        "time":data[6],
        "close_price":data[7],
        "open_price":data[8],
        "pisition":data[9],
        "date":data[12]
    }

#comexData=pd.Series(comex_gold,index=['date','time','real_price','price_change','buy_price','sell_price','high','low','close_price',\
#                                      'open_price','pisition'])

    comexData=pd.DataFrame(comex_silver,columns=['date','time','real_price','price_change','buy_price','sell_price','high','low','close_price',\
                                               'open_price','pisition'],index=[0])
    dt=str(datetime.datetime.now().date())
    #comexData.to_csv("E:\\pythoncoding\\data\\gold\\COMEX_silver_"+dt+".csv")
#以添加模式加入到csv文件中
    comexData.to_csv("E:\\pythoncoding\\data\\gold\\COMEX_silver_"+dt+".csv",mode='a',header=False) 



'''
从新浪行情获取上海金交所 黄金T+D 实时数据
并以添加模式写入到csv中
'''
shanghaigold_TD_Url ="http://hq.sinajs.cn/list=SGE_AUTD"
shanghaigoldTD_web_data = requests.get(shanghaigold_TD_Url)
shanghaigoldTD_soup = BeautifulSoup(shanghaigoldTD_web_data.text,"lxml")
rawdata=shanghaigoldTD_soup.get_text()
rawdataList=rawdata.split("\"")
data = rawdataList[1].split(",")
if len(data)==18:
    shanghai_goldTD={
        "real_price":data[3],
        "price_change":data[4],
        "buy_price":data[10],
        "sell_price":data[11],
        "high":data[7],
        "low":data[8],
        "close_price":data[9],
        "open_price":data[6],
        "pisition":data[14],
        "date":data[16]
    }



    #shanghaiData=pd.Series(shanghai_goldTD,index=['date','real_price','price_change','buy_price','sell_price','high','low','close_price',\
    #                                      'open_price','pisition'])

    shanghaiData=pd.DataFrame(shanghai_goldTD,columns=['date','real_price','price_change','buy_price','sell_price','high','low','close_price',\
                                                'open_price','pisition'],index=[0])
    dt=str(datetime.datetime.now().date())
    #shanghaiData.to_csv("E:\\pythoncoding\\data\\gold\\shanghai_goldTD_"+dt+".csv")
#以添加模式加入到csv文件中
    shanghaiData.to_csv("E:\\pythoncoding\\data\\gold\\shanghai_goldTD_"+dt+".csv",mode='a',header=False)


'''
从新浪行情获取上海金交所 白银T+D 实时数据
并以添加模式写入到csv中
'''
shanghaiSilver_TD_Url ="http://hq.sinajs.cn/list=SGE_AGTD"
shanghaiSilverTD_web_data = requests.get(shanghaiSilver_TD_Url)
shanghaiSilverTD_soup = BeautifulSoup(shanghaiSilverTD_web_data.text,"lxml")
rawdata=shanghaiSilverTD_soup.get_text()
rawdataList=rawdata.split("\"")
data = rawdataList[1].split(",")
if len(data)==18:
    shanghai_silverTD={
        "real_price":data[3],
        "price_change":data[4],
        "buy_price":data[10],
        "sell_price":data[11],
        "high":data[7],
        "low":data[8],
        "close_price":data[9],
        "open_price":data[6],
        "pisition":data[14],
        "date":data[16]
    }



    #shanghaiData=pd.Series(shanghai_silverTD,index=['date','real_price','price_change','buy_price','sell_price','high','low','close_price',\
    #                                      'open_price','pisition'])

    shanghaiData=pd.DataFrame(shanghai_silverTD,columns=['date','real_price','price_change','buy_price','sell_price','high','low','close_price',\
                                                'open_price','pisition'],index=[0])
    dt=str(datetime.datetime.now().date())
    #shanghaiData.to_csv("E:\\pythoncoding\\data\\gold\\shanghai_silverTD_"+dt+".csv")
#以添加模式加入到csv文件中
    shanghaiData.to_csv("E:\\pythoncoding\\data\\gold\\shanghai_silverTD_"+dt+".csv",mode='a',header=False)

    