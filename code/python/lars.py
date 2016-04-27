import numpy as np
import matplotlib.pyplot as plt
from code.python.cubicSpline import CubicSpline
from numba import jit

N = 50
dim = 2
c2 = 1
gamma = 1
density_0 = 10
h = 4#1.45
lx = 0.4 * h
ly = 0.4 * h
m = 1
pos = np.random.random((N, dim))
pos[:, 0] *= lx
pos[:, 1] *= ly
kernel = CubicSpline(dim, h)

vel = np.zeros((N, dim))
acc = np.zeros((N, dim))

dt = 0.004
N_t = 2000
picture_time = 5
g = np.array([0, -1])
mu = 4

@jit
def calc_distances_and_density(positions):
    array_length = len(positions[:, 0])
    density = np.ones(array_length) * m * kernel.kernel(d=0)
    distance_vec = np.zeros((array_length, array_length, dim))
    distance_abs = np.zeros((array_length, array_length))

    for i in range(array_length):
        for j in range(i + 1, array_length):
            distance_vec[i, j] = positions[j] - positions[i]
            distance_abs[i, j] = np.sqrt(np.sum(distance_vec[i, j] ** 2))
            tmp = m * kernel.kernel(d=distance_abs[i, j])
            density[i] += tmp
            density[j] += tmp
    return distance_vec, distance_abs, density

@jit
def calc_acc(distance_vec, distance_abs, density, velocities):
    pressure = density_0 * c2 / gamma * ((density / density_0) ** gamma - 1)
    # print(np.average(pressure))
    array_length = len(density)
    dimensions = len(distance_vec[0, 0, :])
    acceleration = np.zeros((array_length, dimensions))

    for i in range(array_length):
        for j in range(i + 1, array_length):
            if distance_abs[i, j] < 2 * h:
                pres_tmp = pressure[i] / density[i] ** 2 + pressure[j] / density[j] ** 2
                if np.dot(velocities[j] - velocities[i], distance_vec[i, j]) > 0:
                    visc_tmp = 2 * (-c2 * mu + 2 * mu * mu) / (density[i] + density[j])
                else:
                    visc_tmp = 0
                tmp = m * (pres_tmp - visc_tmp) * kernel.gradient(distance_vec[i, j], distance_abs[i, j])
                acceleration[i] -= tmp
                acceleration[j] += tmp
    return acceleration


dis_vec, dis_abs, dens = calc_distances_and_density(pos)


def make_time_step(positions, velocities):
    positions += velocities * dt
    bdry_cdt = np.floor(positions/lx) * np.mod(positions, lx)
    positions -= 2 * bdry_cdt
    distances_vec, distances_abs, density = calc_distances_and_density(positions)
    accelerations = calc_acc(distances_vec, distances_abs, density, velocities)
    velocities += (accelerations + g) * dt
    velocities[bdry_cdt != 0] *= - np.sign(bdry_cdt[bdry_cdt != 0])
    return positions, velocities


for t in range(N_t):
    pos, vel = make_time_step(pos, vel)
    if t % picture_time == 0:
        plt.clf()
        plt.scatter(pos[:, 0], pos[:, 1])
        plt.axis([0, lx, 0, ly])
        plt.savefig("plots/"+str(int(t / picture_time))+".png" )
