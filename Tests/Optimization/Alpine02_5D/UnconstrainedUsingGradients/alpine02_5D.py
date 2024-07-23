import numpy as np
def Alpine02_5D(x):
    prod = 1.0
    for i in range(5):
        prod = prod * np.sqrt(x[i])*np.sin(x[i])
    return -prod
    
def Alpine02_5D_Gradient(x):
    
    fd = np.zeros(5)
    xsave = np.copy(x)
    for i in range(5):
        epsilon = x[i]*0.0001
        
        if(abs(epsilon) < 10E-10):
            epsilon = 0.0001
            x[i] = x[i] + epsilon 
            fp =  Alpine02_5D(x)
            x[i] = xsave[i]
            f  =  Alpine02_5D(x)
            fd[i] = (fp - fm)/(epsilon)
            
        else:
            x[i] = x[i] + epsilon 
            fp =  Alpine02_5D(x)       
            x[i] = x[i] - 2*epsilon    
            fm =  Alpine02_5D(x)
            x[i] = xsave[i]
            fd[i] = (fp - fm)/(2*epsilon)
    
    return fd

