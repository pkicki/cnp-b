import numpy as np
import tensorflow as tf
import pinocchio as pino

from losses.utils import huber
from utils.bspline import BSpline
from utils.manipulator import Iiwa


class FeasibilityLoss:
    def __init__(self, N, urdf_path, q_dot_limits, q_ddot_limits, q_dddot_limits, torque_limits):
        self.bsp_t = BSpline(20)
        self.bsp = BSpline(N)
        self.q_dot_limits = q_dot_limits
        self.q_ddot_limits = q_ddot_limits
        self.q_dddot_limits = q_dddot_limits
        self.torque_limits = torque_limits
        self.iiwa = Iiwa(urdf_path)
        self.model = pino.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()

    @tf.custom_gradient
    def rnea(self, q, dq, ddq):
        torque = np.zeros_like(q)
        n = q.shape[-1]
        dtorque_q = np.zeros(q.shape + (n,))
        dtorque_dq = np.zeros(q.shape + (n,))
        dtorque_ddq = np.zeros(q.shape + (n,))
        q_ = q.numpy() if type(q) is not np.ndarray else q
        dq_ = dq.numpy() if type(dq) is not np.ndarray else dq
        ddq_ = ddq.numpy() if type(ddq) is not np.ndarray else ddq
        q_ = np.pad(q_, ((0, 0), (0, 0), (0, self.model.nq - q_.shape[-1])))
        dq_ = np.pad(dq_, ((0, 0), (0, 0), (0, self.model.nq - dq_.shape[-1])))
        ddq_ = np.pad(ddq_, ((0, 0), (0, 0), (0, self.model.nq - ddq_.shape[-1])))

        def grad(upstream):
            for i in range(q_.shape[0]):
                for j in range(q_.shape[1]):
                    g = pino.computeRNEADerivatives(self.model, self.data, q_[i, j], dq_[i, j], ddq_[i, j])
                    dtorque_q[i, j] = g[0][:n, :n]
                    dtorque_dq[i, j] = g[1][:n, :n]
                    dtorque_ddq[i, j] = g[2][:n, :n]
            g_q = (dtorque_q @ upstream[..., tf.newaxis])[..., 0]
            g_dq = (dtorque_dq @ upstream[..., tf.newaxis])[..., 0]
            g_ddq = (dtorque_ddq @ upstream[..., tf.newaxis])[..., 0]
            return g_q, g_dq, g_ddq
        for i in range(q_.shape[0]):
            for j in range(q_.shape[1]):
                torque[i, j] = pino.rnea(self.model, self.data, q_[i, j], dq_[i, j], ddq_[i, j])[:n]
        return torque, grad

    def call(self, q_cps, t_cps, data):
        q = self.bsp.N @ q_cps
        q_dot_tau = self.bsp.dN @ q_cps
        q_ddot_tau = self.bsp.ddN @ q_cps
        q_dddot_tau = self.bsp.dddN @ q_cps

        dtau_dt = self.bsp_t.N @ t_cps
        ddtau_dtt = self.bsp_t.dN @ t_cps
        dddtau_dttt = self.bsp_t.ddN @ t_cps

        dt = 1. / dtau_dt[..., 0] / dtau_dt.shape[1]
        t_cumsum = np.cumsum(dt, axis=-1)
        t = tf.reduce_sum(dt, axis=-1)

        q_dot = q_dot_tau * dtau_dt
        q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt
        q_dddot = q_dddot_tau * dtau_dt ** 3 + 3 * q_ddot_tau * ddtau_dtt * dtau_dt ** 2 + \
                  q_dot_tau * dtau_dt ** 2 * dddtau_dttt + q_dot_tau * ddtau_dtt ** 2 * dtau_dt

        q_dot_limits = tf.constant(self.q_dot_limits)[tf.newaxis, tf.newaxis]
        q_ddot_limits = tf.constant(self.q_ddot_limits)[tf.newaxis, tf.newaxis]
        q_dddot_limits = tf.constant(self.q_dddot_limits)[tf.newaxis, tf.newaxis]
        torque_limits = tf.constant(self.torque_limits)[tf.newaxis, tf.newaxis]

        torque = self.rnea(q, q_dot, q_ddot)

        torque_loss_ = tf.nn.relu(tf.abs(torque) - torque_limits)
        torque_loss_ = huber(torque_loss_)
        torque_loss = tf.reduce_sum(torque_loss_ * dt[..., tf.newaxis], axis=1)

        q_dot_loss_ = tf.nn.relu(tf.abs(q_dot) - q_dot_limits)
        q_dot_loss_ = huber(q_dot_loss_)
        q_dot_loss = tf.reduce_sum(q_dot_loss_ * dt[..., tf.newaxis], axis=1)
        q_ddot_loss_ = tf.nn.relu(tf.abs(q_ddot) - q_ddot_limits)
        q_ddot_loss_ = huber(q_ddot_loss_)
        q_ddot_loss = tf.reduce_sum(q_ddot_loss_ * dt[..., tf.newaxis], axis=1)
        q_dddot_loss_ = tf.nn.relu(tf.abs(q_dddot) - q_dddot_limits)
        q_dddot_loss_ = huber(q_dddot_loss_)
        q_dddot_loss = tf.reduce_sum(q_dddot_loss_ * dt[..., tf.newaxis], axis=1)
        model_losses = tf.concat([q_dot_loss, q_ddot_loss, q_dddot_loss], axis=-1)
        model_loss = tf.reduce_sum(model_losses, axis=-1)
        return model_loss, q_dot_loss, q_ddot_loss, q_dddot_loss, torque_loss, q, q_dot, q_ddot, q_dddot, torque,\
               t, t_cumsum, dt

    def __call__(self, q_cps, t_cps, data):
        return self.call(q_cps, t_cps, data)
