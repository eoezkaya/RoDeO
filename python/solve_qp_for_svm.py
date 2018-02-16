from cvxopt import matrix, solvers
import sys
import numpy as np


# solution of the convec optimization problem

n_samples = int(sys.argv[2])
 
data = np.genfromtxt(sys.argv[3])
C = float(sys.argv[1])

print 'dimension of the problem = ',n_samples
print 'C = ',C


K = matrix(0.0,(n_samples,n_samples))
p = matrix(-1.0,(n_samples,1))

A = matrix(0.0,(1,n_samples))
b = matrix(0.0)

if(C < 10E10):
	tmp1 = np.diag(np.ones(n_samples) * -1)
	tmp2 = np.identity(n_samples)
	G = matrix(np.vstack((tmp1, tmp2)))
	tmp1 = np.zeros(n_samples)
	tmp2 = np.ones(n_samples) * C
	h = matrix(np.hstack((tmp1, tmp2)))
else:
	print 'here'
	G = matrix(np.diag(np.ones(n_samples) * -1.0))
	h = matrix(np.zeros(n_samples))

	print G
count=0
for i in range(n_samples):
	for j in range(n_samples):
		K[i,j]=data[count]
		count = count+1

for i in range(n_samples):
	A[0,i]=data[count]
	count= count+1

#print G
print A

#print K

#print p

sol=solvers.qp(K, p, G, h, A, b)
print sol
# Lagrange multipliers
a = np.ravel(sol['x'])

np.savetxt('Lagrange_multipliers.txt', a)




