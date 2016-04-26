import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit

# Main constants
Nx = 5                          # Number of particles in x direction
Ny = 5                          # Number of particles in y direction
N = Nx * Ny                     # Total number of particles
h = 0.0260                      # Characteristic width
cut_off1sqrd = 1
cut_off2sqrd = 4
width = 1
height = 1
dimensions = 2
h_dimensions = h**dimensions
sigma = 2/3*(dimensions == 1) + 10/(7*np.pi)*(dimensions == 2) + 1/np.pi*(dimensions == 3)
m = 1

# Initial conditions
initial_positions = np.zeros((N,dimensions))
initial_xs = np.linspace(0, width, Nx)
initial_ys = np.linspace(0, height, Ny)
for i in range(Nx):
    for j in range(Ny):
            initial_positions[i*Ny+j,:] = [initial_xs[i],initial_ys[j]]

@jit
def calc_density(locations):
    W = 0                                           # Smoothing Kernel
    rho = np.ones(N)*sigma/h_dimensions             # Density
    for i in range(N):
        for j in range(i + 1, N):
            # Determine the distances in each spatial direction:
            dx = locations[i, 0] - locations[j, 0]
            dy = locations[i, 1] - locations[j, 1]
            r2 = dx * dx + dy * dy
            q = r2/h**2
            if q <= cut_off1sqrd:
                W = sigma/h_dimensions*(1-1.5*q+0.75*q**(2/3))
            elif q <= cut_off2sqrd:
                 W = sigma/h_dimensions*(0.25*(2-q**(1/2))**3)
            rho[i] += m*W
            rho[j] += m*W
    return rho

density = calc_density(initial_positions)

print(density)

