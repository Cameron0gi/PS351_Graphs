#!/usr/bin/env python
# coding: utf-8

# In[35]:


import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy import integrate
# Define the ODE system dx/dt and dy/dt as functions
def dxdt(x, y):
    return y

def dydt(x, y):
    return -x + (1 - x**2) * y

# Create a grid of points in the x and y space
x_range = np.linspace(-2, 2, 20)
y_range = np.linspace(-2, 2, 20)
x, y = np.meshgrid(x_range, y_range)

# Compute dx/dt and dy/dt at each point in the grid
dx = dxdt(x, y)
dy = dydt(x, y)

# Create a quiver plot to visualize the vector field
plt.figure(figsize=(13.5,5))

plt.subplot(1,3,1)
plt.quiver(x, y, dx, dy, scale=15, color='blue', pivot='middle')
plt.xlabel('x')
plt.ylabel('y')
plt.title('(a)')

plt.subplot(1,3,2)

plt.streamplot(x, y, dx, dy, density=1, color='blue', linewidth=1)
plt.xlabel('x')
plt.ylabel('y')
plt.title('(b)')

plt.subplot(1,3,3)

plt.streamplot(x, y, dx, dy, density=1, color='k', linewidth=1)

irange = np.linspace(-2, 2, 100)
y_isocline = irange / (1 - irange**2)
plt.plot(irange, y_isocline, label='y isocline (dy/dt = 0)', linestyle='--', color='b')

# Plot the y isocline (x / (1 - x^2))
x_isocline = irange / (1 - irange**2)
plt.plot(x_isocline, irange, label='x isocline (dx/dt = 0)', linestyle='--', color='r')
plt.legend(loc='upper right', prop={'size': 9})

plt.xlim(-2,2)
plt.ylim(-2,2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('(c)')









plt.savefig('gm6_1.svg')
plt.show()


# In[1]:


import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

def plot6(x0,y0, filename):
# Define the rabbit and fox population dynamics as a function
    def func(t, z):
        x, y = z
        dxdt = y
        dydt = -x + (1 - x**2) * y
        return [dxdt, dydt]

    # Define the initial conditions


    tf = 30  # Final time
    n = 100  # Number of points at which output will be evaluated
    t_span = (0, tf)

    # Create a figure to plot the solutions
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))


    # Solve the differential equations
    sol = integrate.solve_ivp(
        fun=lambda t, z: func(t, z),
        t_span=t_span,
        y0=[x0, y0],
        t_eval=np.linspace(t_span[0], t_span[1], n),
        method="RK45"
    )

    # Plot the populations of rabbits and foxes
    axs[0].plot(sol.t, sol.y[0], label='x')
    axs[0].plot(sol.t, sol.y[1], label='y')
    axs[0].set_xlabel('Time')
    axs[0].set_ylabel('Population')
    axs[0].legend()
    axs[0].set_xlim(0,30)


    # Phase plot
    axs[1].plot(sol.y[0], sol.y[1])
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')

    # Define the filename for saving the phase plot
  
    plt.savefig(f"{filename}.png")
    plt.show()
      # Save the phase plot to a file


# In[3]:


plot6(3, 3, 'gm6_2a')


# In[2]:


plot6(-2, 2, 'gm6_2b')


# In[9]:


plot6(.1, .1, 'gm6_2c')


# In[10]:


plot6(.5, 1, 'gm6_2d')


# In[15]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def plotgm5(x0, y0):
    def func(t, z):
        x, y = z
        dxdt = y
        dydt = -x + (1 - x**2) * y
        return [dxdt, dydt]
    # Define the initial conditions
    tf = 10  # Final time
    n = 100  # Number of points at which output will be evaluated
    t_span = (0, tf)

    # Solve the differential equations
    sol = integrate.solve_ivp(
        fun=lambda t, z: func(t, z),
        t_span=t_span,
        y0=[x0, y0],
        t_eval=np.linspace(t_span[0], t_span[1], n),
        method="RK45"
    )

    # Plot the populations of rabbits and foxes on the same graph
    plt.plot(sol.y[0], sol.y[1], label=f'x0={x0}, y0={y0}')
    
    


# Example usage with different initial conditions
plotgm5(3, 3)
plotgm5(-2, 2)
plotgm5(0.1, 0.1)
plotgm5(0.5, 1)
plotgm5(1, 1)
plotgm5(4, 3)
# Customize plot labels and legend
plt.xlabel('x')
plt.ylabel('y')
plt.xlim(-3,6)
plt.ylim(-3,4)


irange = np.linspace(-2, 2, 100)
y_isocline = irange / (1 - irange**2)
plt.plot(irange, y_isocline, label='y isocline (dy/dt = 0)', linestyle='--', color='b')

# Plot the y isocline (x / (1 - x^2))
x_isocline = irange / (1 - irange**2)
plt.plot(x_isocline, irange, label='x isocline (dx/dt = 0)', linestyle='--', color='r')
plt.legend(loc='upper right', prop={'size': 9})


# Show the plot
plt.savefig('gm6.svg')
plt.show()


# In[23]:


import matplotlib.pyplot as plt
import numpy as np

# Define the ODE system dx/dt and dy/dt as functions
def dxdt(x, y):
    return y

def dydt(x, y):
    return -x + (1 - x**2) * y

# Create a grid of points in the x and y space
x_range = np.linspace(-2, 2, 20)
y_range = np.linspace(-2, 2, 20)
x, y = np.meshgrid(x_range, y_range)

# Compute dx/dt and dy/dt at each point in the grid
dx = dxdt(x, y)
dy = dydt(x, y)

# Create a quiver plot to visualize the vector field
plt.figure(figsize=(5, 5))
plt.streamplot(x, y, dx, dy, density=1, color='k', linewidth=1)

irange = np.linspace(-2, 2, 100)
y_isocline = irange / (1 - irange**2)
plt.plot(irange, y_isocline, label='y isocline (dy/dt = 0)', linestyle='--', color='b')

# Plot the y isocline (x / (1 - x^2))
x_isocline = irange / (1 - irange**2)
plt.plot(x_isocline, irange, label='x isocline (dx/dt = 0)', linestyle='--', color='r')
plt.legend(loc='upper right', prop={'size': 9})

plt.xlim(-2,2)
plt.ylim(-2,2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Vector Field for dx/dt and dy/dt with isoclines')

plt.show()

