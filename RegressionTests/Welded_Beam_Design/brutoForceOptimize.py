import numpy as np
from welded_beam_optimization import WeldedBeamOptimization

    
    
ub = [2.0, 10.0, 10.0, 2.0]
lb = [0.125, 0.1, 0.1, 0.1 ]
optimal_value = 10**10
func = WeldedBeamOptimization()
for i in range(10000000):
    x = np.random.rand(4)
    x[0] = x[0]*(ub[0]-lb[0])+lb[0]
    x[1] = x[1]*(ub[1]-lb[1])+lb[1]
    x[2] = x[2]*(ub[2]-lb[2])+lb[2]
    x[3] = x[3]*(ub[3]-lb[3])+lb[3]



    f = func.evaluateFunction(x)	
    c1 = func.evaluateConstraint1(x)
    c2 = func.evaluateConstraint2(x)
    c3 = func.evaluateConstraint3(x)
    c4 = func.evaluateConstraint4(x)
    c5 = func.evaluateConstraint5(x)
#    print(x,f,c1,c2,c3,c4,c5)
 
    if(c1 < 0.0 and c2 < 0.0 and c3 > func.P and c4 < func.tauMax and c5 < 0):
        
        if(f<optimal_value):
            print(optimal_value,c1,c2,c3,c4,c5)
            print(x, "\n")
            optimal_value = f
	
	


