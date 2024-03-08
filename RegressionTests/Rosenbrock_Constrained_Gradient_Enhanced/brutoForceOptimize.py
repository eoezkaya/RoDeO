import numpy as np
from rosenbrock_optimization import RosenbrockOptimization

    
    
ub = [1.5, 2.5]
lb = [-1.5, -0.5]
optimal_value = 10**10
rosenbrockOpt = RosenbrockOptimization()
for i in range(100000000):
    x = np.random.rand(2)
    x[0] = x[0]*(ub[0]-lb[0])+lb[0]
    x[1] = x[1]*(ub[1]-lb[1])+lb[1]
    
    f = rosenbrockOpt.evaluateFunction(x)	
    c1 = rosenbrockOpt.evaluateConstraint1(x)
    c2 = rosenbrockOpt.evaluateConstraint2(x)
#    print(x,f,c1,c2)
 
    if(c1 > 0.0 and c2 > 0.0):
        
        if(f<optimal_value):
            print(optimal_value,c1,c2)
            print(x, "\n")
            optimal_value = f
	
	


