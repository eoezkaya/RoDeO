import numpy as np
import matplotlib.pyplot as plt
xlist = np.linspace(-6.0, 6.0, 100)
ylist = np.linspace(-6.0, 6.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = (((X**2+Y-11)**2) + (((X+Y**2-7)**2)))
fig,ax=plt.subplots(1,1)
cp = ax.contourf(X, Y, Z, 50)
fig.colorbar(cp) # Add a colorbar to a plot
ax.set_title('Himmelblau function')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
plt.show()
#plt.savefig('himmelblau.png')
