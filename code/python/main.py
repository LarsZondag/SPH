import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib import cm
from numba import jit

# Main constants
Nx = 7                          # Number of particles in x direction
Ny = 7                          # Number of particles in y direction
N = Nx * Ny                     # Total number of particles
h = 1                           # Characteristic width
width = 0.4
height = 0.4
dimensions = 2
h_dimensions = h**dimensions
h_dimensionplusone = h**(dimensions+1)
sigma = 2/3*(dimensions == 1) + 10/(7*np.pi)*(dimensions == 2) + 1/np.pi*(dimensions == 3)
m = 1
density_0 = 1
c2 = 1
gamma = 1                       # See Jos Thijssen notes (typical value for gamma is 7)
picture_time = 5

# Initial conditions
def initialize():
    initial_positions = np.zeros((N,dimensions))
    initial_xs = np.linspace(0, width, Nx)
    initial_ys = np.linspace(0, height, Ny)
    velos = np.zeros((N,dimensions))
    accel = np.zeros((N,dimensions))
    for i in range(Nx):
        for j in range(Ny):
                initial_positions[i*Ny+j,:] = [initial_xs[i],initial_ys[j]]
    return initial_positions, velos, accel

def calc_kernel(rsqrd):
    W = 0                                                           # Smoothing Kernel
    q2 = rsqrd/h**2
    if q2 <= 1:
        W = sigma/h_dimensions*(1-1.5*q2+0.75*q2**(2/3))
    elif q2 <= 4:
        W = sigma/h_dimensions*(0.25*(2-q2**(1/2))**3)
    return W

def calc_density(locs):
    rho = np.ones(N)*sigma/h_dimensions                         # Density
    grad_W = np.zeros((N,dimensions))
    for i in range(N):
        for j in range(i + 1, N):
            dx = locs[i, 0] - locs[j, 0]
            dy = locs[i, 1] - locs[j, 1]
            r2 = dx * dx + dy * dy
            kernel = calc_kernel(r2)
            rho[i] += m*kernel
            rho[j] += m*kernel
            r = r2**(0.5)
            q = r/h
            if q <= 1:
                common_grad_W = sigma/h_dimensionplusone*((9/4)*q**2-3*q)
                grad_W[i, 0] += common_grad_W*dx/r
                grad_W[i, 1] += common_grad_W*dy/r
                grad_W[j, 0] -= common_grad_W*dx/r
                grad_W[j, 1] -= common_grad_W*dy/r
            elif q <= 2:
                common_grad_W = sigma/h_dimensionplusone*(-0.75*(2-q)**2)
                grad_W[i, 0] += common_grad_W*dx/r
                grad_W[i, 1] += common_grad_W*dy/r
                grad_W[j, 0] -= common_grad_W*dx/r
                grad_W[j, 1] -= common_grad_W*dy/r
    return rho, grad_W

def calc_acceleration(P, rho, gradient_W):
    a = np.zeros((N,dimensions))
    for i in range(N):
        for j in range(i + 1, N):
            a[i] -= m*(P[i]/rho[i]**2 + P[j]/rho[j]**2)*gradient_W[i]
            a[j] += m*(P[i]/rho[i]**2 + P[j]/rho[j]**2)*gradient_W[i]
    return a

def make_time_step(loc, v, old_a):
    loc += v * dt
    v += 0.5 * old_a * dt
    a = calc_acceleration(pressure, density, gradient_kernel)
    v += 0.5 * a * dt
    return loc, a

dt = 0.004
Nt = 500
locations, velocities, acceleration = initialize()
for t in range(Nt):
    density, gradient_kernel = calc_density(locations)
    pressure = density_0*c2 / gamma * ((density/density_0)**gamma-1)
    locations, acceleration = make_time_step(locations, velocities, acceleration)
    if t % picture_time == 0:
        plt.clf()
        plt.scatter(locations[:, 0], locations[:, 1])
        #plt.axis([0, lx, 0, ly])
        plt.savefig("plots/"+str(int(t / picture_time))+".png" )



