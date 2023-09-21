
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [7, 7]
circle1 = plt.Circle((0, 0), 3.16227766, color='r')

fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot
# (or if you have an existing figure)
# fig = plt.gcf()
# ax = fig.gca()

ax.add_patch(circle1)


import numpy as np
import matplotlib.pyplot as plt
xlist = np.linspace(-6.0, 6.0, 100)
ylist = np.linspace(-6.0, 6.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = (((X**2+Y-11)**2) + (((X+Y**2-7)**2)))
plt.contour(X,Y,Z,50)


plt.xlabel('x1')
plt.ylabel('x2') 
plt.title('Himmelblau function')

plt.show()

