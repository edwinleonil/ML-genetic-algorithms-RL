# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Initialize parameters
learning_rate = 0.1
iterations = 50
m = len(X)

# Initial parameters (theta0: intercept, theta1: slope)
theta0, theta1 = np.random.randn(2)

# Gradient Descent
plt.figure(figsize=(10, 6))

for iteration in range(iterations):
    # Predict current values
    y_pred = theta0 + theta1 * X

    # Compute gradients
    grad_theta0 = (2/m) * np.sum(y_pred - y)
    grad_theta1 = (2/m) * np.sum((y_pred - y) * X)

    # Update parameters
    theta0 -= learning_rate * grad_theta0
    theta1 -= learning_rate * grad_theta1

    # Plot the regression line at current iteration
    plt.clf()
    plt.scatter(X, y, label="Training Data")
    plt.plot(X, y_pred, 'r-', linewidth=2, label=f"Iteration {iteration + 1}")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Iterative Linear Regression (Gradient Descent)")
    plt.legend()
    plt.pause(0.5)

plt.show()

# Print final model parameters
print(f"Final intercept (bias): {theta0:.2f}")
print(f"Final coefficient (slope): {theta1:.2f}")
