import numpy as np

optimal_value = 10**10
for i in range(100000000):
    x = np.random.rand(4)
    x[0] = x[0] * 30 + 20.0
    x[1] = x[1] * 9  + 1
    x[2] = x[2] * 30 + 20.0
    x[3] = x[3] * 59.9 + 0.1
	
    f = 8.61*10**5*x[0]**0.5*x[1]*x[2]**(-2/3)*x[3]**(-0.5) + 3.69*10**4*x[2] + 7.72*10**8*(1.0/x[0])*x[1]**0.219 - 765.43*10**6*(1.0/x[0]) 	
    h = x[3]*x[1]**-2 + x[1]**-2 - 1
	
    if(h<0):
        if(f<optimal_value):
            print(optimal_value,x)
            optimal_value = f
	
	


