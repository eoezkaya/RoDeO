import numpy as np
class WeldedBeamOptimization:
    def __init__(self):
        self.L = 14
        self.P = 6000
        self.E = 30000000
        self.sigmaMax = 30000
        self.tauMax = 13600
        self.G = 12000000
        self.deltaMax = 0.25
   
    def evaluateFunction(self, x):
        return 0.04811*x[2]*x[3]*(x[1]+14) + 1.10471*x[0]**2*x[1]        
    def evaluateConstraint1(self,x):
        return x[0] - x[3]
    def evaluateConstraint2(self,x):
        return self.delta(x)- self.deltaMax
        
    def evaluateConstraint3(self,x):
        return self.Pc(x)
    
    def evaluateConstraint4(self,x):
        return self.tau(x)
        
    def evaluateConstraint5(self,x):
        return self.sigma(x)- self.sigmaMax
    

    def tau_prime(self,x):
        return self.P/( np.sqrt(2)*x[0]*x[1])
        
    def R(self,x):
        return np.sqrt(  x[1]**2/4 + ( (x[0] + x[2])/2)**2  )
        
    def J(self,x):
        return 2.0*( (x[1]**2/4 + ((x[0] + x[2])/2)**2)* ( np.sqrt(2)*x[0]*x[1]) )
     
    def delta(self,x):
        return (6*self.P*self.L**3)/(self.E * x[2]**2 * x[3])
        
    def sigma(self,x):
        return (6*self.P*self.L)/(x[2]**2 * x[3])
       
    def Pc(self,x):
        coeff1 = np.sqrt(self.E/(4*self.G))
        coeff2 = (1-x[2]/(2*self.L)*coeff1)
        return (4.013*self.E*x[2]*x[3]**3/(6*self.L**2))*coeff2
    def M(self,x):
        return self.P*(x[1]/2+ self.L)
        
    def tau_prime_prime(self,x):
        return (self.R(x)*self.M(x)) / self.J(x)
        
    def tau(self,x):
        return np.sqrt(self.tau_prime(x)**2 + self.tau_prime_prime(x)**2 + 2*self.tau_prime(x)* self.tau_prime_prime(x) * x[1]/(2*self.R(x)))
