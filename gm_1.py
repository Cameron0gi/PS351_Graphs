#!/usr/bin/env python
# coding: utf-8

# In[85]:


import matplotlib.pyplot as plt
import numpy as np

plt.close()

# Create Grid
coords = np.linspace(-np.pi,np.pi, 40)
x, y = np.meshgrid(coords, coords)

# Create Function 1 and gradient
vx = np.cos(x) * y
vy = np.sin(x) * x


# Create Figure
fig=plt.figure(figsize=(8, 4))
fig.suptitle('Vector function: $V(x,y)=(\cos(x) \cdot y, \sin(x) \cdot x)$', size=14)

plt.subplot(1, 2, 1) # row 1, col 2 index 1
plt.gca().set_aspect('equal', adjustable='box')  # Make plot box square
plt.xlabel('x')
plt.ylabel('y')

plt.quiver(x, y, vx, vy, scale=40)

plt.subplot(1, 2, 2) # row 1, col 2 index 1

plt.gca().set_aspect('equal', adjustable='box')  # Make plot box square
plt.xlabel('x')
plt.ylabel('y')

S=3 #Number of points to skip
x,y,vx,vy=x[::S,::S],y[::S,::S],vx[::S,::S],vy[::S,::S]

plt.quiver(x, y, vx, vy, scale=40)  # Adjust the scale as needed

plt.savefig('gm1.svg')
plt.show()


# In[132]:


import matplotlib.pyplot as plt
import numpy as np
import math

plt.close()

# Create Grid
q = 1
coords = np.linspace(-np.pi, np.pi, 100)
x, y = np.meshgrid(coords, coords)

# Create Function
z = q / np.sqrt((y ** 2 + x ** 2 + 1**-9))

# Calculate the gradient of z
dx = coords[1] - coords[0]  # Grid spacing in x
dy = coords[1] - coords[0]  # Grid spacing in y
dZdx, dZdy = np.gradient(z)

# Create Figure
plt.figure(figsize=(6, 6))
plt.gca().set_aspect('equal', adjustable='box')  # Make plot box square
plt.xlabel('x')
plt.ylabel('y')
plt.title('Scalar function: $z = \\frac{q}{{\sqrt{x^2 + y^2 + \u03B5}}}$')

# Plot scalar function as a color plot (contour map)
contour = plt.contourf(x, y, z, 50, cmap='coolwarm')  # Plot a contour map using N=100 levels
plt.set_cmap('coolwarm')

# Plot Gradient as quiver plot
# Create a coarse grid
S = 5 # Number of points to skip
X, Y, dZdx_coarse, dZdy_coarse = x[::S, ::S], y[::S, ::S], dZdx[::S, ::S], dZdy[::S, ::S]
plt.quiver(X, Y, dZdx_coarse, dZdy_coarse, scale=0.5, color='k')  # Plot the gradient using the coarse grid.

plt.savefig('gm_2.png', bbox_inches='tight')
plt.show()

