import numpy as np
from matplotlib.pyplot import *
from pylab import *
t=linspace(-2*np.pi,2*np.pi,50)
y=np.sin(t)
x=np.cos(t)
xlabel("t")
ylabel("x/y")
xlim(-np.pi,np.pi)
ylim(-1,2)
plot(t,x,'r-')
plot(t,y,'g.')
figtext(0.68,0.65,'y=sint')
show()

