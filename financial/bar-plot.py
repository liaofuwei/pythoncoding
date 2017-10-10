#-*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np  

left = (0,0.3,0.6) #代表柱状图左边起点的位置
height = (2,4,7)   #代表柱子的高度
width=(0.1,0.1,0.1) #代表柱子的宽度
plt.bar(left,height,width) #里面left,height,width顺序不能错

'''XTICKs=(0,0,0)
for i in range(0,2):
    XTICKs[i]=left[i]+(width[i])/2
plt.xticks(XTICKs,('a','b','c'))
元组不能修改，似乎没有统一处理的办法，待定
'''

plt.xticks((0.05,0.35,0.65),('a','b','c'))#前面代表标记的位置
plt.xlim(0,1)
plt.ylim(0,10)
plt.show()

