class RosenbrockOptimization:
   
    def evaluateFunction(self, x):
        return (1-x[0])**2 + 100.0 * (x[1] - x[0]*x[0])**2	        
    def evaluateConstraint1(self,x):
        return x[1] - x[0]*x[0]

