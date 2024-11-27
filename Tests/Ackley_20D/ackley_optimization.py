import numpy as np

class AckleyOptimization:
    
    def __init__(self, dim=20):
        self.dim = dim  # Set the dimension of the function
    
    def evaluateFunction(self, x):
        # Ensure input x matches the expected dimension
        if len(x) != self.dim:
            raise ValueError(f"Input vector must be of dimension {self.dim}.")
        
        # Ackley function parameters
        a = 20
        b = 0.2
        c = 2 * np.pi

        # Calculate the first sum term
        sum_sq_term = np.sum(x**2)
        
        # Calculate the second sum term
        cos_term = np.sum(np.cos(c * x))
        
        # Calculate the Ackley function value
        term1 = -a * np.exp(-b * np.sqrt(sum_sq_term / self.dim))
        term2 = -np.exp(cos_term / self.dim)
        return term1 + term2 + a + np.e
    
    def evaluateGradient(self, x, epsilon=1e-6):
        # Ensure input x matches the expected dimension
        if len(x) != self.dim:
            raise ValueError(f"Input vector must be of dimension {self.dim}.")
        
        # Initialize gradient vector
        gradient = np.zeros(self.dim)
        fx = self.evaluateFunction(x)  # Original function value at x
        
        # Calculate each partial derivative using finite differences
        for i in range(self.dim):
            x_perturbed = np.array(x, dtype=float)  # Copy x to perturb
            x_perturbed[i] += epsilon  # Perturb x[i] by epsilon
            fx_perturbed = self.evaluateFunction(x_perturbed)  # Function value at perturbed x
            # Approximate the partial derivative
            gradient[i] = (fx_perturbed - fx) / epsilon
        
        return gradient

