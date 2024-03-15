class RosenbrockOptimization:
   
    def evaluateFunction(self, x):
        return (1-x[0])**2 + 100.0 * (x[1] - x[0]*x[0])**2	        
    def evaluateConstraint1(self,x):
        return x[1] - x[0]*x[0]
    def evaluateConstraint2(self,x):
        return (x[0] - 1)**3 - x[1] + 0.7   
    def evaluateGradient(self,x):
        dfdx = 2.0* (-1.0 + x[0] + 200*x[0]**3 - 200.0*x[0]*x[1] )
        dfdy = 200.0* (-x[0]**2 + x[1])
        return [dfdx, dfdy]        


