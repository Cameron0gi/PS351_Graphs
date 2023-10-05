#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate


# In[4]:




def rabbit_and_fox(t, z, a, b, c, d):
    x, y = z
    dxdt = a * x - b * x * y
    dydt = c * x * y - d * y
    return [dxdt, dydt]

# Define the initial conditions
x0 = 4  # Initial population of rabbits
y0 = 2  # Initial population of foxes
a = 4  # Rabbit birth rate
b = 2  # Rabbit predation rate
c = 0.33  # Fox reproduction rate
d = 1  # Fox death rate

tf = 10  # Final time
n = 100  # Number of points at which output will be evaluated
t_span = (0, tf)

# Create a figure to plot the solutions
plt.figure()

# Solve the differential equations
sol = integrate.solve_ivp(
    fun=lambda t, z: rabbit_and_fox(t, z, a, b, c, d),
    t_span=t_span,
    y0=[x0, y0],
    t_eval=np.linspace(t_span[0], t_span[1], n),
    method="RK45"
)

# Plot the populations of rabbits and foxes
plt.plot(sol.t, sol.y[0], label='Rabbits')
plt.plot(sol.t, sol.y[1], label='Foxes')

plt.xlabel('Time (t)')
plt.ylabel('Population')
plt.legend()
plt.title('Predator-Prey Model')
plt.savefig("gm5_1.svg")
plt.show()

plt.figure()
plt.plot(sol.y[0], sol.y[1])
plt.xlabel('Rabbits')
plt.ylabel('Foxes')
plt.savefig("gm5_1_phase.svg")


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def plotgm5(x0, y0, a, b, c, d, filename):
    def rabbit_and_fox(t, z, a, b, c, d):
        x, y = z
        dxdt = a * x - b * x * y
        dydt = c * x * y - d * y
        return [dxdt, dydt]

    # Define the initial conditions
    tf = 10  # Final time
    n = 100  # Number of points at which output will be evaluated
    t_span = (0, tf)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Solve the differential equations
    sol = integrate.solve_ivp(
        fun=lambda t, z: rabbit_and_fox(t, z, a, b, c, d),
        t_span=t_span,
        y0=[x0, y0],
        t_eval=np.linspace(t_span[0], t_span[1], n),
        method="RK45"
    )

    # Plot the populations of rabbits and foxes in the first subplot
    axs[0].plot(sol.t, sol.y[0], label='Rabbits')
    axs[0].plot(sol.t, sol.y[1], label='Foxes')
    axs[0].set_xlabel('Time (t)')
    axs[0].set_ylabel('Population')
    axs[0].legend()
    axs[0].set_xlim(0,10)
    # Plot the phase diagram in the second subplot
    axs[1].plot(sol.y[0], sol.y[1])
    axs[1].set_xlabel('Rabbits')
    axs[1].set_ylabel('Foxes')

    # Adjust spacing between subplots
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{filename}.svg")

    # Show the plot
    plt.show()

# Example usage:
# plotgm5(2, 1, 1, 1, 1, 1, "population")


# In[6]:


plotgm5(4, 3, 4, 2, 0.33, 1, 'gm5a1')


# In[7]:


plotgm5(6, 1, 4, 2, 0.33, 1, 'gm5a2')


# In[8]:


plotgm5(1, 6, 4, 2, 0.33, 1, 'gm5a3')


# In[10]:


plotgm5(2, 0, 4, 2, 0.33, 1, 'gm5a4')


# In[11]:


plotgm5(0, 2, 4, 2, 0.33, 1, 'gm5a5')


# In[12]:


plotgm5(4, 2, 2, 4, 1, 0.33, 'gm5b1')


# In[13]:


plotgm5(4, 3, 4, 2, 1, 1, 'gm5b1')


# In[14]:


plotgm5(4, 3, 6, 2, 0.33, 0, 'gm5b2')


# In[15]:


plotgm5(4, 3, 50, 2, 0.33, 1, 'gm5b3')


# In[16]:


plotgm5(4, 3, 4, 2, 0.05, 1, 'gm5b4')

