import numpy as np
class EggholderOptimization:
   
    def evaluateFunction(self, x):
        return -(x[1]+47.0)*np.sin(np.sqrt(abs(x[1]+0.5*x[0]+47.0)))-x[0]*np.sin(np.sqrt(abs(x[0]-(x[1]+47.0) )))	        
    


