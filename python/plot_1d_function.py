import matplotlib.pyplot as plt
import numpy as np
import sys



data = np.genfromtxt(sys.argv[1])
x = data[:,0]
y = data[:,1]

plt.plot(x, y)



plt.show()



exit()

