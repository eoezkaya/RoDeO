from pyDOE import *
import sys
import matplotlib.pyplot as plt

lhs_points_file = sys.argv[1]
dimension = sys.argv[2]
number_of_samples = sys.argv[3]
method = sys.argv[4]

print 'Sampling by LHS Center method'

design = lhs(int(dimension), samples= int(number_of_samples), criterion=method)

print design


x = design[:,0]
y = design[:,1]
print x

plt.plot(x, y, "o")
figure_file = "lhs_points.png"
plt.savefig(figure_file)

f = open('lhs_points.dat', 'w')


for i in range(int(number_of_samples)):
	f.write(str(design[i,0])+' '+str(design[i,1])+'\n')


plt.show(block=False)

f.close()
exit()
