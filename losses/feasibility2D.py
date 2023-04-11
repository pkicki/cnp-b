import numpy as np
import tensorflow as tf

from losses.utils import huber
from utils.bspline import BSpline


class Feasibility2DLoss:
    def __init__(self, N):
        self.bsp_t = BSpline(20)
        self.bsp = BSpline(N)
        self.v_max_sq = np.sqrt(1.)
        self.a_max_sq = np.sqrt(0.5)

    def call(self, xy_cps, t_cps, data):
        xy = self.bsp.N @ xy_cps
        xy_dot_tau = self.bsp.dN @ xy_cps
        xy_ddot_tau = self.bsp.ddN @ xy_cps

        dtau_dt = self.bsp_t.N @ t_cps
        ddtau_dtt = self.bsp_t.dN @ t_cps

        dt = 1. / dtau_dt[..., 0] / dtau_dt.shape[1]
        t_cumsum = np.cumsum(dt, axis=-1)
        t = tf.reduce_sum(dt, axis=-1)

        xy_dot = xy_dot_tau * dtau_dt
        xy_ddot = xy_ddot_tau * dtau_dt ** 2 + ddtau_dtt * xy_dot_tau * dtau_dt

        xy_dot_loss_ = tf.nn.relu(tf.reduce_sum(tf.square(xy_dot), axis=-1) - self.v_max_sq)
        xy_dot_loss_ = huber(xy_dot_loss_)
        xy_dot_loss = tf.reduce_sum(xy_dot_loss_ * dt, axis=1)
        xy_ddot_loss_ = tf.nn.relu(tf.reduce_sum(tf.square(xy_ddot), axis=-1) - self.a_max_sq)
        xy_ddot_loss_ = huber(xy_ddot_loss_)
        xy_ddot_loss = tf.reduce_sum(xy_ddot_loss_ * dt, axis=1)

        model_losses = tf.concat([xy_dot_loss, xy_ddot_loss], axis=-1)
        model_loss = tf.reduce_sum(model_losses, axis=-1)
        return model_loss, xy_dot_loss, xy_ddot_loss, xy, xy_dot, xy_ddot, t, t_cumsum, dt

    def __call__(self, xy_cps, t_cps, data):
        return self.call(xy_cps, t_cps, data)
