from math import pi

import tensorflow as tf
import numpy as np

from utils.constants import Limits, Env
from utils.data import unpack_data_iiwa_obstacles


class IiwaObstaclesPlannerBoundaries(tf.keras.Model):
    def __init__(self, N, n_pts_fixed_begin, n_pts_fixed_end, bsp, bsp_t):
        super(IiwaObstaclesPlannerBoundaries, self).__init__()
        self.N = N - n_pts_fixed_begin - n_pts_fixed_end
        self.n_dof = 7
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
            #tf.keras.layers.Dense(N, activation),
            #tf.keras.layers.Dense(N, activation),
            #tf.keras.layers.Dense(N, activation),
        ]

        self.obstacles = [
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

    def prepare_weights(self):
        self.obstalces_np = []
        for l in self.obstacles:
            self.obstalces_np.append((l.kernel.numpy().T[np.newaxis], l.bias.numpy()[np.newaxis], np.tanh))
        self.fc_np = []
        for l in self.fc:
            self.fc_np.append((l.kernel.numpy().T[np.newaxis], l.bias.numpy()[np.newaxis], np.tanh))
        self.q_np = []
        for l in self.q_est:
            self.q_np.append((l.kernel.numpy().T[np.newaxis], l.bias.numpy()[np.newaxis], np.tanh))
        self.t_np = []
        for l in self.t_est:
            self.t_np.append((l.kernel.numpy().T[np.newaxis], l.bias.numpy()[np.newaxis], np.exp))

    def prepare_data(self, x):
        q0, qd, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d, obstacles = unpack_data_iiwa_obstacles(x)

        expected_time = tf.reduce_max(tf.abs(qd - q0) / Limits.q_dot7[np.newaxis], axis=-1)

        obstacles = tf.reshape(obstacles, (-1, Env.n_obs, 4))

        xb = q0 / pi
        if self.n_pts_fixed_begin > 1:
            xb = tf.concat([xb, q_dot_0 / Limits.q_dot7[np.newaxis]], axis=-1)
        if self.n_pts_fixed_begin > 2:
            xb = tf.concat([xb, q_ddot_0 / Limits.q_ddot7[np.newaxis]], axis=-1)
        xe = qd / pi
        if self.n_pts_fixed_end > 1:
            xe = tf.concat([xe, q_dot_d / Limits.q_dot7[np.newaxis]], axis=-1)
        if self.n_pts_fixed_end > 2:
            xe = tf.concat([xe, q_ddot_d / Limits.q_ddot7[np.newaxis]], axis=-1)

        x = tf.concat([xb, xe], axis=-1)
        return x, q0, qd, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d, obstacles, expected_time

    def prepare_data_inference(self, x):
        q0, qd, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d, obstacles = unpack_data_iiwa_obstacles(x)

        expected_time = np.max(np.abs(qd - q0) / Limits.q_dot7[np.newaxis], axis=-1)

        obstacles = np.reshape(obstacles, (-1, Env.n_obs, 4))

        xb = q0 / pi
        if self.n_pts_fixed_begin > 1:
            xb = np.concatenate([xb, q_dot_0 / Limits.q_dot7[np.newaxis]], axis=-1)
        if self.n_pts_fixed_begin > 2:
            xb = np.concatenate([xb, q_ddot_0 / Limits.q_ddot7[np.newaxis]], axis=-1)
        xe = qd / pi
        if self.n_pts_fixed_end > 1:
            xe = np.concatenate([xe, q_dot_d / Limits.q_dot7[np.newaxis]], axis=-1)
        if self.n_pts_fixed_end > 2:
            xe = np.concatenate([xe, q_ddot_d / Limits.q_ddot7[np.newaxis]], axis=-1)

        x = np.concatenate([xb, xe], axis=-1)
        return x, q0, qd, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d, obstacles, expected_time


    def __call__(self, x, mul=1.):
        x, q0, qd, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d, obstacles, expected_time = self.prepare_data(x)

        for l in self.fc:
            x = l(x)

        for l in self.obstacles:
            obstacles = l(obstacles)

        obstacles = tf.reduce_sum(obstacles, axis=1)

        x = tf.concat([x, obstacles], axis=-1)

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

    def inference(self, x):
        t = []
        # t.append(perf_counter())
        x, q0, qd, q_dot_0, q_dot_d, q_ddot_0, q_ddot_d, obstacles, expected_time = self.prepare_data_inference(x)
        # t.append(perf_counter())
        q_ddot_d = np.zeros_like(q_ddot_0)

        # for l in self.fc:
        #    x = l(x)
        for k, b, a in self.fc_np:
            x = a((k @ x[..., np.newaxis])[..., 0] + b)

        for k, b, a in self.obstalces_np:
            obstacles = a((k @ obstacles[..., np.newaxis])[..., 0] + b)

        obstacles = np.sum(obstacles, axis=1)

        x = np.concatenate([x, obstacles], axis=-1)

        q_est = x
        # for l in self.q_est:
        #    q_est = l(q_est)
        for k, b, a in self.q_np:
            q_est = a((k @ q_est[..., np.newaxis])[..., 0] + b)

        dtau_dt = x
        # for l in self.t_est:
        #    dtau_dt = l(dtau_dt)
        for k, b, a in self.t_np:
            dtau_dt = a((k @ dtau_dt[..., np.newaxis])[..., 0] + b)
        # t.append(perf_counter())

        # q_est = q_est.numpy()
        # dtau_dt = dtau_dt.numpy()
        # t.append(perf_counter())

        # dtau_dt = dtau_dt / expected_time_[:, tf.newaxis]
        dtau_dt = dtau_dt / expected_time[:, np.newaxis]

        # t.append(perf_counter())
        q = pi * np.reshape(q_est, (-1, self.N, self.n_dof))
        # t.append(perf_counter())
        s = np.linspace(0., 1., q.shape[1] + 2)[np.newaxis, 1:-1, np.newaxis]
        # t.append(perf_counter())

        q1 = q_dot_0 / dtau_dt[:, :1] / self.qd1 + q0
        # t.append(perf_counter())
        qm1 = qd - q_dot_d / dtau_dt[:, -1:] / self.qd1
        # t.append(perf_counter())
        q2 = ((q_ddot_0 / dtau_dt[:, :1] -
               self.qd1 * self.td1 * (q1 - q0) * (dtau_dt[:, 1] - dtau_dt[:, 0])[:, np.newaxis]) / dtau_dt[:, :1]
              - self.qdd1 * q0 - self.qdd2 * q1) / self.qdd3
        qm2 = ((q_ddot_d / dtau_dt[:, -1:] -
                self.qd1 * self.td1 * (qd - qm1) * (dtau_dt[:, -1] - dtau_dt[:, -2])[:, np.newaxis]) / dtau_dt[:, -1:]
               - self.qdd1 * qd - self.qdd2 * qm1) / self.qdd3
        # t.append(perf_counter())

        q0 = q0[:, np.newaxis]
        q1 = q1[:, np.newaxis]
        q2 = q2[:, np.newaxis]
        qm1 = qm1[:, np.newaxis]
        qm2 = qm2[:, tf.newaxis]
        qd = qd[:, np.newaxis]

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
        # t.append(perf_counter())

        qb = q_begin[-1] * (1 - s) + q_end[-1] * s

        x = np.concatenate(q_begin + [q + qb] + q_end[::-1], axis=-2)
        # t.append(perf_counter())
        # for i in range(len(t) - 1):
        #    print(i, t[i+1] - t[i])
        return x, dtau_dt[..., np.newaxis]
