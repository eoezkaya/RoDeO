import numpy as np
from rastrigin_optimization import RastriginOptimization

    
ub = 15*[4.12]
lb = 15*[-7.12]


optimal_value = 10**10
func = RastriginOptimization()
for i in range(100000000):
    x = np.random.rand(15)
    for j in range(15):
        x[j] = x[j]*(ub[j]-lb[j])+lb[j]
   
    
    f = func.evaluateFunction(x)	
#    print(x,f)
        
    if(f<optimal_value):
        print(optimal_value)
        print(x, "\n")
        optimal_value = f
	
	


