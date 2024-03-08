import numpy as np
def func(x):    
    return x[0]**2.0*x[1]*(2.0+x[2])

def h1(x):
    return 1 - (x[1]**3*x[2])/(71785*x[0]**4)
def h2(x):
    return (4.0*x[1]**2-x[0]*x[1])/(12566*x[1]*x[0]**3 - x[0]**4) + 1.0/(5108*x[0]**2) - 1.0
def h3(x):
    return 1.0- (140.45*x[0])/(x[1]**2*x[2])
def h4(x):
    return (x[0]+x[1])/1.5 - 1.0



optimal_value = 10**10
for i in range(100000000):
    x = np.random.rand(3)
    x[0] = x[0]*(2-0.05)+0.05
    x[1] = x[1]*(1.3-0.25)+0.25
    x[2] = x[2]*(15-2)+2
    
    f = func(x)	
    c1 = h1(x)
    c2 = h2(x)
    c3 = h3(x)
    c4 = h4(x)
    print(x,f,c1,c2,c3,c4)
 
    if(c1 <= 0.0 and c2 <= 0.0 and c3 <= 0.0 and c4 <= 0.0):
        
        if(f<optimal_value):
            print(optimal_value,c1,c2,c3,c4)
            print(x, "\n")
            optimal_value = f
	
	


