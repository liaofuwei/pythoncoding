from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook
import matplotlib.ticker as ticker
import datetime
import matplotlib.finance as finance

tickerA = '999999.ss'
begdate = datetime.date(1997,1,1)
enddate = datetime.date.today()
priceA = finance.fetch_historical_yahoo(tickerA, begdate, enddate)
rA = mlab.csv2rec(priceA); priceA.close()
rA.sort()

#r = r[-30:]  # get the last 30 days
figA, axA = plt.subplots()
axA.plot(rA.date, rA.adj_close, 'o-')
#ax.set_title('Fig. 1: EBR last 30 days with gaps on weekends')
figA.autofmt_xdate()
N = len(rA)
ind = np.arange(N)  # the evenly spaced plot indices

def format_date(x, pos=None):
    thisind = np.clip(int(x+0.5), 0, N-1)
    return rA.date[thisind].strftime('%Y-%m-%d')

figA, axA = plt.subplots()
axA.plot(ind, rA.adj_close, 'o-')
plt.xlabel("Every Monday shown")
axA.set_title('Fig 2: IBM last 30 days evenly spaced plot indices')
axA.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
figA.autofmt_xdate()
plt.show()
