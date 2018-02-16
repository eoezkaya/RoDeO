import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib.cm as cm

x=[]
y=[]

filename = sys.argv[1]

f = open(filename, "r")
lines = f.read().splitlines() 

f.close()

number_of_clusters =  int(lines[0])

colors = cm.rainbow(np.linspace(0, 1, number_of_clusters))


line_count=1

xcm=[]
ycm=[]


for i in range(number_of_clusters):
	numbers = lines[line_count]
	numbers = lines[line_count].split()
	xcm.append(float(numbers[0]))
	ycm.append(float(numbers[1]))
	line_count=line_count+1 

plt.scatter(xcm,ycm,s=50,color='black')


for i,c in zip(range(number_of_clusters),colors):
	number_of_points = int(lines[line_count])
	line_count=line_count+1 
	for j in range(number_of_points):
		numbers = lines[line_count]
		numbers = lines[line_count].split()
#		print numbers
		x.append(float(numbers[0]))
		y.append(float(numbers[1]))
		line_count=line_count+1 
	plt.scatter(x,y,color=c)
	x=[]
	y=[]	


plt.show()

exit()
