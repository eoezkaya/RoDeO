import matplotlib.pyplot as plt
import numpy as np
import sys
import pandas

function_name = sys.argv[1]
filename = function_name +"_TestResults.csv"
savePlotFileName = function_name +"_TestResults.png"

df = pandas.read_csv(filename)
data = df.values
problemDimension = df.shape[1]-3


fExact = data[:,problemDimension]
fSurrogate = data[:,problemDimension+1]
SE = data[:,problemDimension+2]
MSE = np.mean(SE)
textstr="Mean Squared Error = "+str(MSE)

fig, ax = plt.subplots(figsize=(10,7))
ax.set_title(function_name,fontsize=22)
ax.set_xlabel("true function", fontsize=18)
ax.set_ylabel("surrogate", fontsize=18)
ax.scatter(fExact,fSurrogate)
x1, y1 = [min(fExact), max(fExact)], [min(fExact), max(fExact)]

ax.plot(x1, y1, color = "red")
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

# place a text box in upper left in axes coords
ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
        verticalalignment='top', bbox=props)
plt.savefig(savePlotFileName)
plt.show()

exit()

