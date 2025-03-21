import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import FancyArrowPatch

# Define the quadratic function: y = x^2


def f(x):
    return x**2

# Define the derivative (gradient) of the function: dy/dx = 2x


def gradient(x):
    return 2 * x


# Gradient descent parameters
learning_rate = 0.2   # step size
iterations = 50       # number of iterations
x_current = 8         # initial guess for x

# Lists to store values for plotting
x_history = [x_current]
y_history = [f(x_current)]

# Gradient descent loop
for i in range(iterations):
    grad = gradient(x_current)
    x_current = x_current - learning_rate * grad  # update rule
    x_history.append(x_current)
    y_history.append(f(x_current))
    print(f"Iteration {i+1}: x = {x_current:.4f}, f(x) = {f(x_current):.4f}")

# Plot the gradient descent progress in real time
x = np.linspace(-10, 10, 400)
y = f(x)

fig, ax = plt.subplots()
ax.plot(x, y, label='y = x²')
scat = ax.scatter([], [], color='red', label='Gradient Descent Path')
line, = ax.plot([], [], 'r-', lw=2)  # Line to show the gradient
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Gradient Descent Example')
ax.legend()
ax.grid(True)

# Initialize text annotation
text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

# Initialize arrow annotation
arrow = FancyArrowPatch((0, 0), (0, 0), mutation_scale=25,
                        color='blue', arrowstyle='->')
ax.add_patch(arrow)


def update(frame):
    scat.set_offsets(np.c_[x_history[:frame+1], y_history[:frame+1]])
    text.set_text(f'x = {x_history[frame]:.4f}, f(x) = {y_history[frame]:.4f}')

    # Update the gradient line
    x_val = x_history[frame]
    y_val = y_history[frame]
    grad = gradient(x_val)
    line.set_data([x_val, x_val + 1], [y_val, y_val + grad])

    # Update the arrow to show the direction of the gradient
    arrow.set_positions(
        (x_val, y_val), (x_val - grad * 0.1, y_val - grad * 0.1))

    return scat, text, line, arrow


ani = FuncAnimation(fig, update, frames=range(
    len(x_history)), blit=True, repeat=False, interval=500)
plt.show()
