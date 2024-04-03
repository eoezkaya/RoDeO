import numpy as np
class Mishra05Optimization:
   
    def evaluateFunction(self, x):
        sumX = x[0]+x[1]
        sumSin = np.sin(x[0]) + np.sin(x[1])
        sumCos = np.cos(x[0]) + np.cos(x[1])
        
        val = ( np.sin(sumCos*sumCos)**2 + np.cos(sumSin*sumSin)**2 + x[0])**2 + 0.01*sumX
        return val
