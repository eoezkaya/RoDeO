import matplotlib.pyplot as plt
import numpy as np
import sys



data = np.genfromtxt(sys.argv[1])
x = data[:,0]
y = data[:,1]
z = data[:,2]

points = np.genfromtxt("lin_reg_points.dat")

xp =points[:,0]
yp = points[:,1]

line_up,=plt.plot(x, y,label='regression')
line_down,=plt.plot(x, z,label='true function')
plt.ylabel('x')
plt.xlabel('y')
plt.legend([line_up, line_down], ['regression', 'true function'])
plt.scatter(xp, yp)


plt.show()



exit()

