import numpy as np
class Mishra03Optimization:
   
    def evaluateFunction(self, x):
        sumX = x[0]+x[1]
        sumSqr = x[0]*x[0] + x[1]*x[1] 
        
        val = np.sqrt(abs(np.cos(np.sqrt(abs(sumSqr))))) + 0.01*sumX;
        return val
