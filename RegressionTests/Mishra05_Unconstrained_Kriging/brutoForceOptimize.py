import numpy as np
from mishra05_optimization import Mishra05Optimization

    
    
ub = [10.0, 10.0]
lb = [-10.0, -10.0]
optimal_value = 10**10
func = Mishra05Optimization()
for i in range(100000000):
    x = np.random.rand(2)
    x[0] = x[0]*(ub[0]-lb[0])+lb[0]
    x[1] = x[1]*(ub[1]-lb[1])+lb[1]
    
    f = func.evaluateFunction(x)	
#    print(x,f)
        
    if(f<optimal_value):
        print(optimal_value)
        print(x, "\n")
        optimal_value = f
	
	


