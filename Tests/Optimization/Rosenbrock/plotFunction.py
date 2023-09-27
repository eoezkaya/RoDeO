import numpy as np
import matplotlib.pyplot as plt
xlist = np.linspace(-1.5, 1.5, 100)
ylist = np.linspace(-0.5, 2.5, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = (1-X)**2 + 100*(Y-X**2)**2
plt.contour(X,Y,Z,50)
plt.xlabel('x1')
plt.ylabel('x2') 
plt.title('Rosenbrock function')
plt.show()

