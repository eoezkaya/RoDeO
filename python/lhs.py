import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas

filename = sys.argv[1]
df = pandas.read_csv(filename,header=None)
data = df.values
x = data[:,0]
y = data[:,1]
plt.scatter(x,y)
plt.show()
exit()
