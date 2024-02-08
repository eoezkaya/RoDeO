import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('optimizationHistory.csv')
dataMatrix = df.to_numpy()
x1 = dataMatrix[:,0]
x2 = dataMatrix[:,1]
f = dataMatrix[:,2]
plt.rcParams["figure.figsize"] = (10,10)
xlist = np.linspace(-6.0, 6.0, 100)
ylist = np.linspace(-6.0, 6.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = (((X**2+Y-11)**2) + (((X+Y**2-7)**2)))
#plt.contour(X,Y,Z,20)
plt.contour(X,Y,Z,[1,10,100, 200, 300, 400], colors = 'black')
plt.scatter(x1, x2, c=f)
plt.scatter(x1, x2, color = 'black')
#plt.xlabel('x1')
#plt.ylabel('x2') 
#plt.colorbar()
#plt.title('Himmelblau function')
#plt.show()
plt.savefig("logo1.png")
