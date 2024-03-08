import numpy as np
from himmelblau_optimization import HimmelblauOptimization

    
    
ub = [6.0, 6.0]
lb = [-6.0, -6.0]
optimal_value = 10**10
himmelblauOpt = HimmelblauOptimization()
for i in range(100000000):
    x = np.random.rand(2)
    x[0] = x[0]*(ub[0]-lb[0])+lb[0]
    x[1] = x[1]*(ub[1]-lb[1])+lb[1]
    
    f = himmelblauOpt.evaluateFunction(x)	
#    print(x,f)
        
    if(f<optimal_value):
        print(optimal_value)
        print(x, "\n")
        optimal_value = f
	
	


