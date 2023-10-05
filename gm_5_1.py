#!/usr/bin/env python
# coding: utf-8

# In[26]:


import matplotlib.pyplot as plt
import numpy as np

# Create Grid for the first plot
coords1 = np.linspace(-1, 5, 20)
x1, y1 = np.meshgrid(coords1, coords1)

# Constants for the first predator-prey model
a = 4  # Rabbit birth rate
b = 3  # Rabbit predation rate
c = 0.33  # Fox reproduction rate
d = 1  # Fox death rate
x0=4
y0=3

# Equations for the first predator-prey model
dx1 = a * x1 - b * x1 * y1
dy1 = c * x1 * y - d * y1

# Create Grid for the second plot
coords2 = np.linspace(-5, 5, 50)
x2, v2 = np.meshgrid(coords2[::2], coords2[::2])


U2 = a * x2 - b * x2 * v2
V2 = c * x2 * v2 - d2* v2

# Create a single figure with subplots
plt.figure(figsize=(12, 6))

# Plot the first vector field using quiver
plt.subplot(121)
plt.gca().set_aspect('equal', adjustable='box')
plt.xlabel('fox')
plt.ylabel('rabbit')
plt.ylim(-0.5, 4)
plt.xlim(-0.5, 3)
plt.title('Predator-Prey Model (1)')
plt.quiver(x1, y1, dx1, dy1, scale=50, pivot='middle', color='blue')

# Plot the second vector field using quiver
plt.subplot(122)
plt.gca().set_aspect('equal', adjustable='box')  # Make plot box square
plt.xlabel('Rabbit')
plt.ylabel('Fox')
plt.title('Predator-Prey Model (2)')
plt.quiver(x2, v2, U2, V2, scale=150)  # plot field as quiver
plt.xlim(0, 5)
plt.ylim(0, 5)

Mag2 = np.sqrt(U2 * U2 + V2 * V2)
lw2 = 30 * Mag2 / Mag2.max()
starting_point2 = np.array([[x0, y0]])  # Specify the starting point
strm2 = plt.streamplot(x2, v2, U2, V2, start_points=starting_point2, linewidth=lw2, color=Mag2, cmap='gnuplot')
plt.colorbar(strm2.lines, fraction=0.046, pad=0.04)

# Show the combined plot
plt.tight_layout()
plt.show()


# In[41]:


import matplotlib.pyplot as plt
import numpy as np

def plotgm5(a, b, c, d, x0, y0,s, name):
    coords1 = np.linspace(-1, 5, 20)
    x1, y1 = np.meshgrid(coords1, coords1)

    # Equations for the first predator-prey model
    dx1 = a * x1 - b * x1 * y1
    dy1 = c * x1 * y - d * y1

    # Create Grid for the second plot
    coords2 = np.linspace(-5, 5, 50)
    x2, v2 = np.meshgrid(coords2[::3], coords2[::3])


    U2 = a * x2 - b * x2 * v2
    V2 = c * x2 * v2 - d2* v2

    # Create a single figure with subplots
    plt.figure(figsize=(12, 6))

    # Plot the first vector field using quiver
    plt.subplot(121)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('Rabbit')
    plt.ylabel('Fox')
    plt.ylim(-0.5, 4)
    plt.xlim(-0.5, 3)
    plt.title('Predator-Prey Model (1)')
    plt.quiver(x1, y1, dx1, dy1, scale=s, pivot='middle', color='blue')

    # Plot the second vector field using quiver
    plt.subplot(122)
    plt.gca().set_aspect('equal', adjustable='box')  # Make plot box square
    plt.xlabel('Rabbit')
    plt.ylabel('Fox')
    plt.title('Predator-Prey Model (2)')
    plt.quiver(x2, v2, U2, V2, scale=s)  # plot field as quiver
    plt.xlim(0, 5)
    plt.ylim(0, 5)

    Mag2 = np.sqrt(U2 * U2 + V2 * V2)
    lw2 = 30 * Mag2 / Mag2.max()
    starting_point2 = np.array([[x0, y0]])  # Specify the starting point
    strm2 = plt.streamplot(x2, v2, U2, V2, start_points=starting_point2, linewidth=lw2, color=Mag2, cmap='gnuplot')
    plt.colorbar(strm2.lines, fraction=0.046, pad=0.04)

    # Show the combined plot
    plt.tight_layout()
    plt.savefig(f'{name}.svg')
    plt.show()


# In[42]:


plotgm5(6, 1, 4, 2, 0.33, 1, 100, 'gm5c2')


# In[43]:


plotgm5(1, 6, 4, 2, 0.33, 1, 200, 'gm5c3')


# In[44]:


plotgm5(2, 0, 4, 2, 0.33, 1, 100, 'gm5c4')


# In[45]:


plotgm5(0, 2, 4, 2, 0.33, 1, 100, 'gm5c5')

