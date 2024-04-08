import numpy as np
from michalewitz_optimization import MichalewitzOptimization

    
ub = 10*[3.141592]
lb = 10*[0.0]


optimal_value = 10**10
func = MichalewitzOptimization()
for i in range(100000000):
    x = np.random.rand(10)
    for j in range(10):
        x[j] = x[j]*(ub[j]-lb[j])+lb[j]
   
    
    f = func.evaluateFunction(x)	
#    print(x,f)
        
    if(f<optimal_value):
        print(optimal_value)
        print(x, "\n")
        optimal_value = f
	
	


