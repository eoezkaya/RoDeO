import numpy as np
import matplotlib.pyplot as plt
xlist = np.linspace(-2.0, 2.0, 100)
ylist = np.linspace(-2.0, 2.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = (1-X)**2 + 100*(Y-X**2)**2
plt.contour(X,Y,Z,50)
plt.xlabel('x1')
plt.ylabel('x2') 
plt.title('Rosenbrock function')
plt.show()

