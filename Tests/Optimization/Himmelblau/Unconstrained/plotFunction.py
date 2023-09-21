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
plt.savefig('himmelblau.png')
