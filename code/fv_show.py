import numpy as np
from matplotlib.pyplot import *
from pylab import *
pv=1000
r=0.08
t=linspace(0,10,10)
fv=pv*(1+r)**t
plot(t,fv)
show()
