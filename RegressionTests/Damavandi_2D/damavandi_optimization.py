import numpy as np
class DamavandiOptimization:
   
    def evaluateFunction(self, x):
        x1 = x[0]
        x2 = x[1]
        numerator = np.sin(np.pi*(x1 - 2.0))*np.sin(np.pi*(x2 - 2.0))
        denumerator = (np.pi**2)*(x1 - 2.0)*(x2 - 2.0)
        factor1 = 1.0 - (abs(numerator / denumerator))**5.0
        factor2 = 2 + (x1 - 7.0)**2.0 + 2*(x2 - 7.0)**2.0

        return factor1*factor2      
   


