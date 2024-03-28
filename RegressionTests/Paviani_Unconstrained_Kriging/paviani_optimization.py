import numpy as np
class PavianiOptimization:
   
    def evaluateFunction(self, x):
        sumTerm = 0
        mul = 1
    
        for i in range(10):
            a = np.log(x[i]-2)
            b = np.log(10.0 - x[i])
            sumTerm += a**2 + b**2
            mul = mul*x[i]
        
        return sumTerm - mul**0.2        
