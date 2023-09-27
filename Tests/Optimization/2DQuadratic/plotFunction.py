import numpy as np
import matplotlib.pyplot as plt
xlist = np.linspace(-30.0, 30.0, 100)
ylist = np.linspace(-30.0, 30.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = 0.5*X**2 + Y**2 - X*Y + 2*X + 6*Y
plt.contour(X,Y,Z,50)
plt.xlabel('x1')
plt.ylabel('x2') 
plt.title('2D Quadratic function')
plt.show()
plt.savefig('2DQuadratic.png')
