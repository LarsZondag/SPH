import numpy as np
import matplotlib.pyplot as plt
from code.python.cubicSpline import CubicSpline

N = 5
m = 1
dim = 2
c2 = 1
gamma = 1
h = 1
density_0 = 1
lx = 0.3 * h
ly = 0.3 * h
pos = np.random.random((N, dim))
pos[:, 0] *= lx
pos[:, 1] *= ly
kernel = CubicSpline(dim, h)

vel = np.zeros((N, dim))
acc = np.zeros((N, dim))

dt = 0.004
N_t = 1000
picture_time = 5


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


def calc_acc(distance_vec, distance_abs, density):
    pressure = density_0 * c2 / gamma * ((density / density_0) ** gamma - 1)
    array_length = len(density)
    dimensions = len(distance_vec[0, 0, :])
    acceleration = np.zeros((array_length, dimensions))

    for i in range(array_length):
        for j in range(i + 1, array_length):
            tmp = m * (pressure[i] / density[i] ** 2 + pressure[j] / density[j] ** 2) * kernel.gradient(distance_vec[i, j], distance_abs[i, j])
            acceleration[i] -= tmp
            acceleration[j] += tmp
    return acceleration


dis_vec, dis_abs, dens = calc_distances_and_density(pos)


def make_time_step(positions, velocities, accelerations):
    distances_vec, distances_abs, density = calc_distances_and_density(positions)
    velocities += 0.5 * accelerations * dt
    positions += velocities * dt
    accelerations = calc_acc(distances_vec, distances_abs, density)
    velocities += 0.5 * accelerations * dt

    return positions, velocities, accelerations


for t in range(N_t):
    pos, vel, acc = make_time_step(pos, vel, acc)
    if t % picture_time == 0:
        plt.clf()
        plt.scatter(pos[:, 0], pos[:, 1])
        plt.axis([0, lx, 0, ly])
        plt.savefig("plots/"+str(int(t / picture_time))+".png" )
