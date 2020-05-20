from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.mlab import griddata
import numpy as np
import sys
import pandas


fig = plt.figure()
ax = fig.gca(projection='3d')

function_name = sys.argv[1]
filename = function_name +"_FunctionPlot.csv"


df = pandas.read_csv(filename,header=None)

data = df.values
x = data[:,0]
y = data[:,1]
z = data[:,2]


xi = np.linspace(min(x), max(x))
yi = np.linspace(min(y), max(y))

X, Y = np.meshgrid(xi, yi)
Z = griddata(x, y, z, xi, yi,interp='linear')

title = function_name

surf = ax.plot_surface(X, Y, Z, rstride=5, cstride=5, cmap=cm.jet,
                       linewidth=1, antialiased=True)

ax.set_zlim3d(np.min(Z), np.max(Z))
ax.set_title(title)
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("f")
fig.colorbar(surf)

figure_file = function_name + "_FunctionPlot.png"

plt.savefig(figure_file)

plt.show()



#key= raw_input("Press any key to continue...\n")
exit()

