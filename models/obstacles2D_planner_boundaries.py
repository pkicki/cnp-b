from math import pi

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils.constants import Limits
from utils.data import unpack_data_boundaries, unpack_data_kinodynamic, unpack_data_acrobot, \
    unpack_data_boundaries_heights, unpack_data_obstacles2D
from utils.normalize import normalize_xy


class Obstacles2DPlannerBoundaries(tf.keras.Model):
    def __init__(self, N, n_pts_fixed_begin, n_pts_fixed_end, bsp, bsp_t):
        super(Obstacles2DPlannerBoundaries, self).__init__()
        self.N = N - n_pts_fixed_begin - n_pts_fixed_end
        self.n_pts_fixed_begin = n_pts_fixed_begin
        self.n_pts_fixed_end = n_pts_fixed_end
        self.xydd1 = bsp.ddN[0, 0, 0]
        self.xydd2 = bsp.ddN[0, 0, 1]
        self.xydd3 = bsp.ddN[0, 0, 2]
        self.xyd1 = bsp.dN[0, 0, 1]
        self.td1 = bsp_t.dN[0, 0, 1]

        activation = tf.keras.activations.tanh
        N = 256
        self.fc = [
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
        ]

        self.xy_est = [
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(self.N * 2, activation),
        ]

        self.t_est = [
            tf.keras.layers.Dense(20, tf.math.exp, name="time_est"),
        ]



    def prepare_data(self, x):
        xy0, xyk, dxy0, dxyk, obstacles = unpack_data_obstacles2D(x)

        ddxy0 = tf.zeros_like(xy0)
        ddxyk = tf.zeros_like(xyk)

        xb = xy0
        if self.n_pts_fixed_begin > 1:
            xb = tf.concat([xb, dxy0], axis=-1)
        if self.n_pts_fixed_begin > 2:
            xb = tf.concat([xb, ddxy0], axis=-1)

        xe = xyk
        if self.n_pts_fixed_end > 1:
            xe = tf.concat([xe, dxyk], axis=-1)
        if self.n_pts_fixed_end > 2:
            xe = tf.concat([xe, ddxyk], axis=-1)

        x = tf.concat([xb, xe, obstacles], axis=-1)
        return x, xy0, xyk, dxy0, dxyk, ddxy0, ddxyk

    def __call__(self, x):
        x, xy0, xyk, dxy0, dxyk, ddxy0, ddxyk = self.prepare_data(x)

        for l in self.fc:
            x = l(x)

        xy_est = x
        for l in self.xy_est:
            xy_est = l(xy_est)

        dtau_dt = x
        for l in self.t_est:
            dtau_dt = l(dtau_dt)

        xy = tf.reshape(xy_est, (-1, self.N, 2))
        s = tf.linspace(0., 1., tf.shape(xy)[1] + 2)[tf.newaxis, 1:-1, tf.newaxis]

        xy1 = dxy0 / dtau_dt[:, :1] / self.xyd1 + xy0
        xym1 = xyk - dxyk / dtau_dt[:, -1:] / self.xyd1
        xy2 = ((ddxy0 / dtau_dt[:, :1] -
               self.xyd1 * self.td1 * (xy1 - xy0) * (dtau_dt[:, 1] - dtau_dt[:, 0])[:, np.newaxis]) / dtau_dt[:, :1]
              - self.xydd1 * xy0 - self.xydd2 * xy1) / self.xydd3
        xym2 = ((ddxyk / dtau_dt[:, -1:] -
              self.xyd1 * self.td1 * (xyk - xym1) * (dtau_dt[:, -1] - dtau_dt[:, -2])[:, np.newaxis]) / dtau_dt[:, -1:]
             - self.xydd1 * xyk - self.xydd2 * xym1) / self.xydd3

        xy0 = xy0[:, tf.newaxis]
        xy1 = xy1[:, tf.newaxis]
        xy2 = xy2[:, tf.newaxis]
        xym1 = xym1[:, tf.newaxis]
        xym2 = xym2[:, tf.newaxis]
        xyk = xyk[:, tf.newaxis]

        xy_begin = [xy0]
        if self.n_pts_fixed_begin > 1:
            xy_begin.append(xy1)
        if self.n_pts_fixed_begin > 2:
            xy_begin.append(xy2)
        xy_end = [xyk]
        if self.n_pts_fixed_end > 1:
            xy_end.append(xym1)
        if self.n_pts_fixed_end > 2:
           xy_end.append(xym2)

        xyb = xy_begin[-1] * (1 - s) + xy_end[-1] * s

        x = tf.concat(xy_begin + [xy + xyb] + xy_end[::-1], axis=-2)
        return x, dtau_dt[..., tf.newaxis]
