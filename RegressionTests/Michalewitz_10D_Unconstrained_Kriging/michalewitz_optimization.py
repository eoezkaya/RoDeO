import numpy as np
class MichalewitzOptimization:
   
    def evaluateFunction(self, x):
    
        dim = 10
        m = 10.0
        pi = 3.141592
        sum = 0.0
        
        for i in range(dim):
            xi = x[i]
            xi_fac = np.sin((i+1)*xi*xi/pi)
            sum = sum + np.sin(xi) * xi_fac**(2*m)
        
        val = -sum
        return val
        
    
