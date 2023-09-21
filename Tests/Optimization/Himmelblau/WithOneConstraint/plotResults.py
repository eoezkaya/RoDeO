import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('optimizationHistory.csv')
dataMatrix = df.to_numpy()
x1 = dataMatrix[:,0]
x2 = dataMatrix[:,1]
f = dataMatrix[:,2]

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
plt.scatter(x1, x2, c=f)

plt.xlabel('x1')
plt.ylabel('x2') 
plt.title('Himmelblau function')

plt.show()



