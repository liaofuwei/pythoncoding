import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cbook as cbook
import matplotlib.ticker as ticker

#data file pathway
fileA_path=r'E:\study\python\EBR_data\EBR-from2008.csv'

datafile = cbook.get_sample_data(fileA_path,asfileobj = False)
print ( 'loading %s' %datafile)
r = mlab.csv2rec(datafile)
r.sort()
#r = r[-30:] # get the last 30 days

# first we 'll do it the default way, with gaps on weekends
fig, ax = plt.subplots()
ax.plot(r.date, r.adj_close, 'o-')
fig.autofmt_xdate()

# next we' ll write a custom formatter
N = len(r)
ind = np.arange(N) # the evenly spaced plot indices
def format_date(x, pos=None):
    thisind = np.clip(int(x+0.5),0,N-1)
    return r.date[thisind].strftime('%Y-%m-%d')

fig, ax = plt.subplots()
ax.plot(ind,r.adj_close,'o-')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(format_date))
fig.autofmt_xdate()
plt.show()
