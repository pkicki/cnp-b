import numpy as np
import tensorflow as tf
import pinocchio as pino

from losses.feasibility import FeasibilityLoss
from losses.utils import huber
from utils.manipulator import Iiwa


class KinodynamicLoss(FeasibilityLoss):
    def __init__(self, N, urdf_path, end_effector_constraint_distance_function, obstacle_distance_function, q_dot_limits, q_ddot_limits,
                 q_dddot_limits, torque_limits):
        super(KinodynamicLoss, self).__init__(N, urdf_path, q_dot_limits, q_ddot_limits, q_dddot_limits, torque_limits)
        self.end_effector_constraints_distance_function = end_effector_constraint_distance_function
        self.obstacle_distance_function = obstacle_distance_function
        self.man = Iiwa(urdf_path)
        self.model = pino.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.alpha_vertical = tf.math.log(1e0)
        self.alpha_constraint = tf.math.log(1e0)
        self.alpha_q_dot = tf.math.log(1e0)
        self.alpha_q_ddot = tf.math.log(1e0)
        self.alpha_torque = tf.math.log(1e0)
        self.gamma = 1e-2
        self.bar_vertical = 1e-5
        self.bar_constraint = 1e-6
        self.bar_q_dot = 6e-3
        self.bar_q_ddot = 6e-2
        self.bar_q_dddot = 6e-1
        self.bar_torque = 6e-2
        self.time_mul = 1e0

    def call(self, q_cps, t_cps, data):
        huber_along_path = lambda x: tf.reduce_sum(dt * huber(x), axis=-1)

        _, q_dot_loss, q_ddot_loss, q_dddot_loss, torque_loss, q, q_dot, q_ddot, q_dddot, torque, t, t_cumsum, dt = super().call(q_cps, t_cps, data)
        links_poses, R = self.man.interpolated_forward_kinematics(q)
        constraint_loss = self.end_effector_constraints_distance_function(links_poses[..., 0], R, dt, data)
        obstacle_loss = np.zeros_like(constraint_loss)
        t_loss = huber(t[:, tf.newaxis])
        vertical_loss = huber_along_path(1.0 - R[:, :, 2, 2])[..., tf.newaxis]
        jerk_loss = tf.reduce_sum(tf.abs(q_dddot) * dt[..., tf.newaxis], axis=(1, 2))[:, tf.newaxis]
        int_torque_loss = tf.reduce_sum(tf.abs(torque) * dt[..., tf.newaxis], axis=(1, 2))[:, tf.newaxis]
        losses = tf.concat([tf.exp(self.alpha_q_dot) * q_dot_loss,
                            tf.exp(self.alpha_q_ddot) * q_ddot_loss,
                            tf.exp(self.alpha_torque) * torque_loss,
                            tf.exp(self.alpha_constraint) * constraint_loss,
                            tf.exp(self.alpha_vertical) * vertical_loss,
                            self.time_mul * t_loss], axis=-1)
        unscaled_losses = tf.concat([q_dot_loss, q_ddot_loss, constraint_loss, obstacle_loss, t_loss], axis=-1)
        sum_q_dot_loss = tf.reduce_sum(q_dot_loss, axis=-1)
        sum_q_ddot_loss = tf.reduce_sum(q_ddot_loss, axis=-1)
        sum_q_dddot_loss = tf.reduce_sum(q_dddot_loss, axis=-1)
        sum_constraint_loss = tf.reduce_sum(constraint_loss, axis=-1)
        sum_torque_loss = tf.reduce_sum(torque_loss, axis=-1)

        model_loss = tf.reduce_sum(losses, axis=-1)
        unscaled_model_loss = tf.reduce_sum(unscaled_losses, axis=-1)
        return model_loss, sum_constraint_loss, sum_q_dot_loss, sum_q_ddot_loss, sum_q_dddot_loss, sum_torque_loss, vertical_loss, \
               q, q_dot, q_ddot, q_dddot, torque, links_poses, t, t_cumsum, t_loss, dt, unscaled_model_loss, jerk_loss, int_torque_loss, constraint_loss

    def alpha_update(self, q_dot_loss, q_ddot_loss, q_dddot_loss, constraint_loss, torque_loss, vertical_loss):
        max_alpha_update = 10.0
        alpha_q_dot_update = self.gamma * tf.clip_by_value(tf.math.log(q_dot_loss / self.bar_q_dot), -max_alpha_update, max_alpha_update)
        alpha_q_ddot_update = self.gamma * tf.clip_by_value(tf.math.log(q_ddot_loss / self.bar_q_ddot), -max_alpha_update, max_alpha_update)
        alpha_constraint_update = self.gamma * tf.clip_by_value(tf.math.log(constraint_loss / self.bar_constraint), -max_alpha_update, max_alpha_update)
        alpha_torque_update = self.gamma * tf.clip_by_value(tf.math.log(torque_loss / self.bar_torque), -max_alpha_update, max_alpha_update)
        alpha_vertical_update = self.gamma * tf.clip_by_value(tf.math.log(vertical_loss / self.bar_vertical), -max_alpha_update, max_alpha_update)
        self.alpha_q_dot += alpha_q_dot_update
        self.alpha_q_ddot += alpha_q_ddot_update
        self.alpha_constraint += alpha_constraint_update
        self.alpha_torque += alpha_torque_update
        self.alpha_vertical += alpha_vertical_update
