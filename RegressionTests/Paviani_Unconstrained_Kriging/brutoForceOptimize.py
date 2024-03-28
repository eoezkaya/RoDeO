import numpy as np
from paviani_optimization import PavianiOptimization

    
    
ub = [9.999, 9.999,9.999,9.999,9.999,9.999,9.999,9.999,9.999,9.999]
lb = [2.001, 2.001,2.001,2.001,2.001,2.001,2.001,2.001,2.001,2.001]
optimal_value = 10**10
func = PavianiOptimization()
for i in range(100000000):
    x = np.random.rand(10)
    x[0] = x[0]*(ub[0]-lb[0])+lb[0]
    x[1] = x[1]*(ub[1]-lb[1])+lb[1]
    x[2] = x[2]*(ub[2]-lb[2])+lb[2]
    x[3] = x[3]*(ub[3]-lb[3])+lb[3]
    x[4] = x[4]*(ub[4]-lb[4])+lb[4]
    x[5] = x[5]*(ub[5]-lb[5])+lb[5]
    x[6] = x[6]*(ub[6]-lb[6])+lb[6]
    x[7] = x[7]*(ub[7]-lb[7])+lb[7]
    x[8] = x[8]*(ub[8]-lb[8])+lb[8]
    x[9] = x[9]*(ub[9]-lb[9])+lb[9]
    
    f = func.evaluateFunction(x)	
#    print(x,f)
        
    if(f<optimal_value):
        print(optimal_value)
        print(x, "\n")
        optimal_value = f
	
	


