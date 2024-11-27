import numpy as np

def cola(x):
    # Given distances
    dis = [
        1.27,
        1.69, 1.43,
        2.04, 2.35, 2.43,
        3.09, 3.18, 3.26, 2.85,
        3.20, 3.22, 3.27, 2.88, 1.55,
        2.86, 2.56, 2.58, 2.59, 3.12, 3.06,
        3.17, 3.18, 3.18, 3.12, 1.31, 1.64, 3.00,
        3.21, 3.18, 3.18, 3.17, 1.70, 1.36, 2.95, 1.32,
        2.38, 2.31, 2.42, 1.94, 2.85, 2.81, 2.56, 2.91, 2.97
    ]

    sum = 0.0
    k = 1
    mt = np.zeros(20)
    # Map x[0] to x[16] to mt[4] to mt[19]
    mt[4:] = x[:16]

    for i in range(1, 10):
        for j in range(i):
            temp = 0.0
            for t in range(2):
                temp += (mt[i*2 + t] - mt[j*2 + t]) ** 2
            sum += (dis[k-1] - np.sqrt(temp)) ** 2
            k += 1
    return sum

# Set bounds for each variable (Assuming bounds based on problem context)
# Let's assume each variable in x can range from -5 to 5
bounds = [(-4, 4) for _ in range(16)]  # x has 16 variables

num_iterations = 1000000  # Number of random samples to evaluate

# Variables to keep track of the best found solution
best_x = None
best_fx = float('inf')  # Start with a high initial best value

# Perform random search
for i in range(num_iterations):
    # Generate a random point within bounds for each variable
    x = np.array([np.random.uniform(low, high) for low, high in bounds])

    # Evaluate the function at this point
    fx = cola(x)

    # Update the best found solution if this one is better
    if fx < best_fx:
        best_fx = fx
        best_x = x

        # Optionally, print progress
        print(f"Iteration {i}, Current Best Function Value: {best_fx}")

# Print the final result
print("\nMinimum found at:")
for idx, val in enumerate(best_x):
    print(f"x[{idx}] = {val:.6f}")
print(f"\nFunction value at minimum: {best_fx:.6f}")

