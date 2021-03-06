import numpy as np
import matplotlib.pyplot as plt
from code.python.cubicSpline import CubicSpline
from numba import jit


dim = 2
c2 = 1
gamma = 7
h = 1
kernel = CubicSpline(dim, h)
m = 4
density_0 = 2
lx = 25
ly = 25
N_x = 20
N_y = 20
N = N_x * N_y
pos = np.zeros((N_x * N_y, dim))
randoms = (np.random.random((N, dim)) - 0.5) * 2 * 0.1
posrender = np.zeros((N_x * N_y, 3))

for i in range(N_x):
    for j in range(N_y):
        pos[i * N_y + j, 0] = i + randoms[i * N_y + j, 0]
        pos[i * N_y + j, 1] = j + 2 + randoms[i * N_y + j, 1]
pos[:,0] /= N_x
pos[:,1] /= N_y

m = np.ones(N)
for i in range(N_x):
    m[N_x*(N_x-1)+i] = 1

pos[:, 0] *= 0.5 * lx
pos[:, 1] *= 0.5 * ly
pos[:, 0] += 0.25 * lx
# pos[:, 1] += 0.1 * ly
v_0 = 10
vel = np.zeros((N, dim))
acc = np.zeros((N, dim))

dt = 0.01
N_t = 4000
picture_time = 10
g = np.array([0, -1])
mu = 1
nu = 1

@jit
def calc_distances_and_density(positions):
    array_length = len(positions[:, 0])
    density = np.ones(array_length) * m[1] * kernel.kernel(d=0)
    distance_vec = np.zeros((array_length, array_length, dim))
    distance_abs = np.zeros((array_length, array_length))

    for i in range(array_length):
        for j in range(i + 1, array_length):
            distance_vec[i, j] = positions[j] - positions[i]
            distance_abs[i, j] = np.sqrt(np.sum(distance_vec[i, j] ** 2))
            tmp = m[j] * kernel.kernel(d=distance_abs[i, j])
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
                if np.dot(velocities[j] - velocities[i], distance_vec[i, j]) < 0:
                    visc_tmp = 2 * (-c2 * mu + 2 * mu * mu) / (density[i] + density[j])
                else:
                    visc_tmp = 0
                tmp = - m[j] * (pres_tmp + visc_tmp) * kernel.gradient(distance_vec[i, j], distance_abs[i, j])
                acceleration[i] += tmp
                acceleration[j] -= tmp
    acceleration -= nu * velocities
    return acceleration


dis_vec, dis_abs, dens = calc_distances_and_density(pos)


def make_time_step(positions, velocities):
    positions += velocities * dt
    bdry_cdt = np.floor(positions/lx) * np.mod(positions, lx)
    positions -= 2 * bdry_cdt
    distances_vec, distances_abs, density = calc_distances_and_density(positions)
    # print(np.average(density))
    accelerations = calc_acc(distances_vec, distances_abs, density, velocities)
    velocities += (accelerations + g) * dt
    velocities[bdry_cdt != 0] *= - np.sign(bdry_cdt[bdry_cdt != 0])
    return positions, velocities


for t in range(N_t):
    pos, vel = make_time_step(pos, vel)
    posrender[:,:2] = pos
    if t % picture_time == 0:
        plt.clf()
        plt.scatter(pos[:90, 0], pos[:90, 1], color='blue')
        plt.scatter(pos[90:, 0], pos[90:, 1], color='blue')
        plt.axis([0, lx, 0, ly])
        plt.savefig("plots/"+str(int(t / picture_time))+".png" )

        np.savetxt("data/"+str(int(t / picture_time))+".txt",posrender[:,:],'%10.5f',",",",")
