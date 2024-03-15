import numpy as np
from eggholder_optimization import EggholderOptimization

    
    
ub = [512, 512]
lb = [-512, -512]
optimal_value = 10**10
eggholderFunc = EggholderOptimization()
for i in range(100000000):
    x = np.random.rand(2)
    x[0] = x[0]*(ub[0]-lb[0])+lb[0]
    x[1] = x[1]*(ub[1]-lb[1])+lb[1]
    
    f = eggholderFunc.evaluateFunction(x)	

#    print(x,f,c1,c2)
 
        
    if(f<optimal_value):
        print(optimal_value)
        print(x, "\n")
        optimal_value = f
	
	


