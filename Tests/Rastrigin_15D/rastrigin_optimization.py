import numpy as np
class RastriginOptimization:
   
    def evaluateFunction(self, x):
        sum = 0.0;
        for i in range(15):
            sum = sum + x[i]**2 - 10.0*np.cos(2*3.141592*x[i])
        val = 10*15 + sum
        return val
