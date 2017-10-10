import datetime
import matplotlib.financeA as finance
import matplotlib.mlab as mlab
import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt

tickerA='000001.ss'
begdateA=datetime.date(2006,11,21)
enddateA=datetime.date(2007,10,11)

begdateB=datetime.date(2014,5,21)
enddateB=datetime.date.today()

priceA=finance.fetch_historical_yahoo(tickerA,begdateA,enddateA)
rA=mlab.csv2rec(priceA);
rA.sort()

priceB=finance.fetch_historical_yahoo(tickerA,begdateB,enddateB)
rB=mlab.csv2rec(priceB);
rB.sort()


for i in range(len(rA)):
    rB[i][0]=rA[i][0]
    




#print rA[0:len(rA)]
fig,ax=plt.subplots()
ax.plot(rA.date,rA.close,'o-')
ax.plot(rB.date,rB.close,'*-')
fig.autofmt_xdate()


"""
N=len(rA)
ind=np.arange(N)

def format_date(x,pos=None):
    thisind=np.clip(int(x+5),0,N-1)
    return rA.date[thisind].strftime('%Y-%m-%d')


fig,ax=plt.subplots()
ax.plot(ind,rA.adj_close,'o-')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
fig.autofmt_xdate()
"""
plt.show()
