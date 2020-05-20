import matplotlib.pyplot as plt
import numpy as np
import sys



data = np.genfromtxt(sys.argv[1])
x = data[:,0]
y = data[:,1]

title = sys.argv[3]
figure_file = sys.argv[2]

fig, ax = plt.subplots()
ax.set_title(title)
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
ax.plot(x, y)

plt.savefig(figure_file)
plt.show()


exit()

