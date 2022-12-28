from math import pi

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from utils.constants import Limits
from utils.data import unpack_data_boundaries, unpack_data_kinodynamic, unpack_data_acrobot
from utils.normalize import normalize_xy


class IiwaPlannerBoundaries(tf.keras.Model):
    def __init__(self, N, n_pts_fixed_begin, n_pts_fixed_end, bsp, bsp_t, n_dof=6):
        super(IiwaPlannerBoundaries, self).__init__()
        self.N = N - n_pts_fixed_begin - n_pts_fixed_end
        self.n_dof = n_dof
        self.n_pts_fixed_begin = n_pts_fixed_begin
        self.n_pts_fixed_end = n_pts_fixed_end
        self.qdd1 = bsp.ddN[0, 0, 0]
        self.qdd2 = bsp.ddN[0, 0, 1]
        self.qdd3 = bsp.ddN[0, 0, 2]
        self.qd1 = bsp.dN[0, 0, 1]
        self.td1 = bsp_t.dN[0, 0, 1]

        activation = tf.keras.activations.tanh
        N = 2048
        self.fc = [
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(N, activation),
        ]

        self.q_est = [
            tf.keras.layers.Dense(N, activation),
            tf.keras.layers.Dense(self.n_dof * self.N, activation),
        ]

        self.t_est = [
            tf.keras.layers.Dense(20, tf.math.exp, name="time_est"),
        ]

    def prepare_data(self, x):
        return NotImplementedError

    def __call__(self, x, mul=1.):
        x, q0, qd, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d, expected_time = self.prepare_data(x)

        for l in self.fc:
            x = l(x)

        q_est = x
        for l in self.q_est:
            q_est = l(q_est)

        dtau_dt = x
        for l in self.t_est:
            dtau_dt = l(dtau_dt)

        dtau_dt = dtau_dt / expected_time[:, tf.newaxis]

        q = pi * tf.reshape(q_est, (-1, self.N, self.n_dof))
        s = tf.linspace(0., 1., tf.shape(q)[1] + 2)[tf.newaxis, 1:-1, tf.newaxis]

        q1 = q_dot_0 / dtau_dt[:, :1] / self.qd1 + q0
        qm1 = qd - q_dot_d / dtau_dt[:, -1:] / self.qd1
        q2 = ((q_ddot_0 / dtau_dt[:, :1] -
               self.qd1 * self.td1 * (q1 - q0) * (dtau_dt[:, 1] - dtau_dt[:, 0])[:, np.newaxis]) / dtau_dt[:, :1]
              - self.qdd1 * q0 - self.qdd2 * q1) / self.qdd3
        qm2 = ((q_ddot_d / dtau_dt[:, -1:] -
              self.qd1 * self.td1 * (qd - qm1) * (dtau_dt[:, -1] - dtau_dt[:, -2])[:, np.newaxis]) / dtau_dt[:, -1:]
             - self.qdd1 * qd - self.qdd2 * qm1) / self.qdd3

        q0 = q0[:, tf.newaxis]
        q1 = q1[:, tf.newaxis]
        q2 = q2[:, tf.newaxis]
        qm1 = qm1[:, tf.newaxis]
        qm2 = qm2[:, tf.newaxis]
        qd = qd[:, tf.newaxis]

        q_begin = [q0]
        if self.n_pts_fixed_begin > 1:
            q_begin.append(q1)
        if self.n_pts_fixed_begin > 2:
            q_begin.append(q2)
        q_end = [qd]
        if self.n_pts_fixed_end > 1:
            q_end.append(qm1)
        if self.n_pts_fixed_end > 2:
           q_end.append(qm2)

        qb = q_begin[-1] * (1 - s) + q_end[-1] * s

        x = tf.concat(q_begin + [q + qb] + q_end[::-1], axis=-2)
        return x, dtau_dt[..., tf.newaxis]


class IiwaPlannerBoundariesHitting(IiwaPlannerBoundaries):
    def __init__(self, N, n_pts_fixed_begin, n_pts_fixed_end, bsp, bsp_t):
        super(IiwaPlannerBoundariesHitting, self).__init__(N, n_pts_fixed_begin, n_pts_fixed_end, bsp, bsp_t)

    def prepare_data(self, x):
        q0, qd, xyth, q_dot_0, q_dot_d, q_ddot_0, puck_pose = unpack_data_boundaries(x, self.n_dof + 1)

        expected_time = tf.reduce_max(tf.abs(qd - q0) / Limits.q_dot[np.newaxis], axis=-1)

        xb = q0 / pi
        if self.n_pts_fixed_begin > 1:
            xb = tf.concat([xb, q_dot_0 / Limits.q_dot[np.newaxis]], axis=-1)
        if self.n_pts_fixed_begin > 2:
            xb = tf.concat([xb, q_ddot_0 / Limits.q_ddot[np.newaxis]], axis=-1)

        xe = qd / pi
        if self.n_pts_fixed_end > 1:
            xe = tf.concat([xe, q_dot_d / Limits.q_dot[np.newaxis]], axis=-1)

        x = tf.concat([xb, xe], axis=-1)
        return x, q0, qd, q_dot_0, q_dot_d, q_ddot_0, np.zeros_like(q_ddot_0), expected_time


class IiwaPlannerBoundariesKinodynamic(IiwaPlannerBoundaries):
    def __init__(self, N, n_pts_fixed_begin, n_pts_fixed_end, bsp, bsp_t):
        super(IiwaPlannerBoundariesKinodynamic, self).__init__(N, n_pts_fixed_begin, n_pts_fixed_end, bsp, bsp_t, n_dof=7)

    def prepare_data(self, x):
        q0, qd, xyz0, xyzk, q_dot_0, q_dot_d, q_ddot_0 = unpack_data_kinodynamic(x, self.n_dof)

        expected_time = tf.reduce_max(tf.abs(qd - q0) / Limits.q_dot7[np.newaxis], axis=-1)

        xb = q0 / pi
        if self.n_pts_fixed_begin > 1:
            xb = tf.concat([xb, q_dot_0 / Limits.q_dot7[np.newaxis]], axis=-1)
        if self.n_pts_fixed_begin > 2:
            xb = tf.concat([xb, q_ddot_0 / Limits.q_ddot7[np.newaxis]], axis=-1)
        xe = qd / pi
        if self.n_pts_fixed_end > 1:
            xe = tf.concat([xe, q_dot_d / Limits.q_dot7[np.newaxis]], axis=-1)

        x = tf.concat([xb, xe], axis=-1)
        return x, q0, qd, q_dot_0, q_dot_d, q_ddot_0, np.zeros_like(q_ddot_0), expected_time