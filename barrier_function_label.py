import numpy as np
import matplotlib.pyplot as plt

N=100
x1= np.linspace(0, 10, N, endpoint=True)
x2= np.linspace(10, 10, N, endpoint=True)
x3= np.linspace(10, 0, N, endpoint=True)
x4= np.linspace(0, 0, N, endpoint=True)

y1=np.linspace(10, 10, N, endpoint=True)
y2=np.linspace(10, 0, N, endpoint=True)
y3=np.linspace(0, 0, N, endpoint=True)
y4=np.linspace(0, 10, N, endpoint=True)


x_total=np.concatenate((x1, x2, x3, x4), axis=0)
y_total=np.concatenate((y1, y2, y3, y4), axis=0)
xy_total=zip(x_total,y_total)
xy_total=np.array(list(xy_total))

N2=100
xy_test=[]
x_test=np.linspace(-2,12, N, endpoint=True)
y_test=np.linspace(-2,12,N, endpoint=True)
for x in x_test:
    for y in y_test:
        xy=(x,y)
        xy_test.append(xy)




print(xy_test,"xy")
'''
plt.plot(x_total,y_total)
plt.ylabel('')
plt.show()
'''

