#!/usr/bin/env python
# coding: utf-8

# In[23]:


import matplotlib.pyplot as plt
import numpy as np
plt.close('all')
coords = np.linspace(-np.pi, np.pi, 21)
x, v = np.meshgrid(coords, coords)

w=1.5

U = v
V = (-w**2)*x


fig=plt.figure(figsize=(13, 6))
fig.suptitle('Harmonic Oscillator: $ d^2x/dt^2+w^2x=0 $', size=18)

plt.subplot(1, 2, 1) # row 1, col 2 index 1
plt.gca().set_aspect('equal', adjustable='box') #Make plot box square
plt.xlabel('x')
plt.ylabel('v')
plt.xlim(-3,3)
plt.streamplot(x,v,U,V, density = [0.5, 0.5])
plt.quiver(x, v, U, V, scale=100) # plot field as quiver

plt.subplot(1, 2, 2)

plt.xlabel('x')
plt.ylabel('v')
plt.xlim(-3,3)
plt.quiver(x, v, U, V) # plot field as quiver
seed_points = np.array([[-2, 1, 2], [-2, 1, 2]])

Mag=np.sqrt(U*U+V*V)
lw=5*Mag/Mag.max()
strm=plt.streamplot(x, v, U, V, linewidth=lw, color=Mag, cmap='gnuplot')
plt.colorbar(strm.lines, fraction=0.046, pad=0.04)




plt.savefig('gm3_1.svg')
plt.show()


# In[25]:


import matplotlib.pyplot as plt
import numpy as np
plt.close('all')
coords = np.linspace(-np.pi, np.pi, 21)
x, v = np.meshgrid(coords, coords)

w=1.5
w=1.5
b=1.7
A=1
wd=1


U = v
V = -b*v-(w**2)*x


fig=plt.figure(figsize=(13, 6))
fig.suptitle('Damped Harmonic Oscillator: $ d^2x/dt^2+ bdx/dt +w^2x=0 $', size=18)

plt.subplot(1, 2, 1) # row 1, col 2 index 1
plt.gca().set_aspect('equal', adjustable='box') #Make plot box square
plt.xlabel('x')
plt.ylabel('v')
plt.xlim(-3,3)
plt.streamplot(x,v,U,V, density = [0.5, 0.5])
plt.quiver(x, v, U, V, scale=100) # plot field as quiver

plt.subplot(1, 2, 2)

plt.xlabel('x')
plt.ylabel('v')
plt.xlim(-3,3)
plt.quiver(x, v, U, V) # plot field as quiver
seed_points = np.array([[-2, 1, 2], [-2, 1, 2]])

Mag=np.sqrt(U*U+V*V)
lw=5*Mag/Mag.max()
strm=plt.streamplot(x, v, U, V, linewidth=lw, color=Mag, cmap='gnuplot')
plt.colorbar(strm.lines, fraction=0.046, pad=0.04)




plt.savefig('gm3_2.svg')
plt.show()

