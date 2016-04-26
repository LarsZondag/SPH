import numpy as np


class CubicSpline(object):
    def __init__(self, dim=1, h=1):
        self.dim = dim
        self.h = h
        self.h1 = 1 / h

        if dim == 3:
            self.fac = 1 / (np.pi * h ** 3)
        elif dim == 2:
            self.fac = 10 / (7 * np.pi * h ** 2)
        else:
            self.fac = 2 / (3 * h)

    def kernel(self, d=1.0):

        u = d / self.h

        umin2 = 2 - u
        if u > 2.0:
            val = 0.0
        elif u > 1.0:
            val = 0.25 * umin2 * umin2 * umin2
        else:
            val = 1 - 1.5 * u * u * (1 - 0.5 * u)

        return val * self.fac

    def gradient(self, r=[0., 0, 0], d=1.0):
        u = d * self.h1
        if 0 <= u <= 1:
            tmp = self.h1 * self.h1 * (9 / 4 * u**2 - 3 * u)
        elif 1 < u <= 2:
            tmp = self.h1 * self.h1 * (- 3 / 4 * (2 - u)**2)
        else:
            tmp = 0
        return self.fac * tmp * r / d