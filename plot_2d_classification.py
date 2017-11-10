import matplotlib.pyplot as plt
import numpy as np
import sys
import csv

data = np.genfromtxt(sys.argv[1])
x = data[:,0]
y = data[:,1]

xmin=[]
ymin=[]
xplus=[]
yplus=[]

if(len(sys.argv) == 4):

	with open(sys.argv[2], 'rb') as csvfile:
		data = csv.reader(csvfile)
		for row in data:
#			print row[2]
			if( float(row[2]) == 1.0):
				xplus.append(float(row[0]))
				yplus.append(float(row[1]))
			if( float(row[2]) == -1.0):
				xmin.append(float(row[0]))
				ymin.append(float(row[1]))

	minus = plt.scatter(xmin, ymin, c='r')
	plus  = plt.scatter(xplus, yplus, c='b')

	true = plt.plot(x, y,label="true function")

	plt.legend([minus, plus, true], ['-1', '+1', 'true function'])
	plt.savefig(sys.argv[3])
	plt.show()

if(len(sys.argv) == 5):
	data = np.genfromtxt(sys.argv[4])
	xsvm = data[:,0]
	ysvm = data[:,1]

	with open(sys.argv[2], 'rb') as csvfile:
		data = csv.reader(csvfile)
		for row in data:
#			print row[2]
			if( float(row[2]) == 1.0):
				xplus.append(float(row[0]))
				yplus.append(float(row[1]))
			if( float(row[2]) == -1.0):
				xmin.append(float(row[0]))
				ymin.append(float(row[1]))

	minus = plt.scatter(xmin, ymin, c='r', label='-')
	plus  = plt.scatter(xplus, yplus, c='b', label='+')
	
	true = plt.plot(x, y,label="true function")
	svm  = plt.plot(xsvm, ysvm, label="svm classifier")
	plt.legend()
	plt.savefig(sys.argv[3])
	plt.show()

exit()
