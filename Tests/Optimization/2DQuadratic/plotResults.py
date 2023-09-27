import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('optimizationHistory.csv')
dataMatrix = df.to_numpy()
x1 = dataMatrix[:,0]
x2 = dataMatrix[:,1]
f = dataMatrix[:,2]


fig, ax = plt.subplots() # note we must use plt.subplots, not plt.subplot




import numpy as np
import matplotlib.pyplot as plt
xlist = np.linspace(-30.0, 30.0, 100)
xline = np.linspace(-10.0, 30.0, 200)
ylist = np.linspace(-30.0, 30.0, 100)
X, Y = np.meshgrid(xlist, ylist)
Z = 0.5*X**2 + Y**2 - X*Y + 2*X + 6*Y

plt.contour(X,Y,Z,50)
plt.scatter(x1, x2, c=f)
plt.plot(xline,20-xline)
plt.xlabel('x1')
plt.ylabel('x2') 
plt.title('2D Quadratic function')

plt.show()



