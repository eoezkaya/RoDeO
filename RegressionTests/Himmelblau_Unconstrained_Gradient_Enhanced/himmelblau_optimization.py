class HimmelblauOptimization:
   
    def evaluateFunction(self, x):
         return (((x[0]**2+x[1]-11)**2) + (((x[0]+x[1]**2-7)**2)))
    def evaluateGradient(self,x):
        temp1 = (x[0]**2+x[1]-11)
        temp2 = (x[0]+x[1]**2-7)
        dfdx1 = 4*temp1*x[0] + 2*temp2
        dfdx2 = 2*temp1      + 4*temp2*x[1]
        return [dfdx1,dfdx2]    



