#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import matplotlib.pyplot as plt

# Define the logistic growth equation
def logistic_growth(x, r, K):
    return r * x * (1 - x / K)

# Define the range of x values
 # Adjust the range as needed

# Choose values for r and K
r = 2  # Adjust as needed
K = 150  # Adjust as needed

x_values = np.linspace(0, 150, 100) 

# Calculate x' for each x value
x_prime = logistic_growth(x_values, r, K)


# Create the plot
fig=plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
r = 2  # Adjust as needed
K = 150  # Adjust as needed

x_values = np.linspace(0, 150, 100) 

# Calculate x' for each x value
x_prime = logistic_growth(x_values, r, K)
plt.plot(x_values, x_prime,label=f'r={r}, K={K}')
plt.xlabel('x')
plt.ylabel("x'")
plt.xlim(0, 150)

plt.legend(loc='upper right')
plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
plt.axvline(x=0, color='k', linestyle='--', linewidth=1)

plt.subplot(1, 2, 2)
r = 2  # Adjust as needed
K = 75  # Adjust as needed

x_values = np.linspace(0, 150, 100) 

# Calculate x' for each x value
x_prime = logistic_growth(x_values, r, K)
plt.plot(x_values, x_prime,label=f'r={r}, K={K}')
plt.xlabel('x')
plt.ylabel("x'")
plt.xlim(0, 150)

plt.legend()
plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
plt.axvline(x=0, color='k', linestyle='--', linewidth=1)

plt.savefig('gm_4.svg')
plt.show()


# In[28]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

def logistic_growth(t, x, r, K):  # Change the order of the arguments
    return r * x * (1 - x / K)

# Define the initial conditions
x0 = [0.2, 5]  # Initial population
r_values = [2, 0.5]
k_values = [2, 1]
tf = 15  # Final time

n = 101  # Number of points at which output will be evaluated
t = np.linspace(0, tf, n)  # Linearly spaced time intervals

# Create a figure to plot the solutions
plt.figure()

for x0 in x0:
    for r in r_values:
        for k in k_values:
            sol = integrate.solve_ivp(fun=lambda t, x: logistic_growth(t, x, r, k), t_span=(0, tf), y0=[x0], t_eval=t, method="RK45")
            y = sol.y[0]

            # Plot the solution
            plt.plot(sol.t, y, label=f'r={r}, K={k}')

plt.xlabel('t')
plt.ylabel("x")
plt.xlim(0, tf)
plt.legend()
plt.savefig("logistic_growth.svg")
plt.show()


# In[56]:


import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
import random 

def logistic_growth(t, x, r, K):
    return r * x * (1 - x / K)

# Define the initial conditions

r_values = [2, 0.5]
k_values = [3,5]
tf = 15  # Final time

n = 101  # Number of points at which output will be evaluated
t = np.linspace(0, tf, n)  # Linearly spaced time intervals

# Create a figure to plot the solutions
plt.figure()

for r in r_values:
    for k in k_values:
        x0=random.uniform(0.5, 2.5)
        sol = integrate.solve_ivp(fun=lambda t, x: logistic_growth(t, x, r, k), t_span=(0, tf), y0=[x0], t_eval=t, method="RK45")
        y = sol.y[0]

        # Plot the solution
        plt.plot(sol.t, y, label=f'r={r}, K={k}')

plt.xlabel('t')
plt.ylabel("x")
plt.xlim(0, tf)
plt.ylim(0,5.5)
plt.legend()
plt.savefig("logistic_growth.svg")
plt.show()

