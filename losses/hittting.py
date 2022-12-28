import tensorflow as tf

from losses.feasibility import FeasibilityLoss
from losses.utils import huber
from utils.manipulator import Iiwa


class HittingLoss(FeasibilityLoss):
    def __init__(self, N, urdf_path, end_effector_constraint_distance_function, obstacle_distance_function, q_dot_limits, q_ddot_limits,
                 q_dddot_limits, torque_limits):
        super(HittingLoss, self).__init__(N, urdf_path, q_dot_limits, q_ddot_limits, q_dddot_limits, torque_limits)
        self.end_effector_constraints_distance_function = end_effector_constraint_distance_function
        self.obstacle_distance_function = obstacle_distance_function
        self.man = Iiwa(urdf_path)
        self.alpha_obstacle = tf.math.log(1e-0)
        self.alpha_constraint = tf.math.log(1e-0)
        self.alpha_q_dot = tf.math.log(1e-2)
        self.alpha_q_ddot = tf.math.log(1e-4)
        self.alpha_q_dddot = tf.math.log(1e-2)
        self.alpha_torque = tf.math.log(1e-2)
        self.gamma = 1e-2
        self.bar_obstacle = 1e-9
        self.bar_constraint = 2e-6
        self.bar_q_dot = 6e-3
        self.bar_q_ddot = 6e-2
        self.bar_q_dddot = 6e-1
        self.bar_torque = 6e-1
        self.centrifugal_mul = 1e-2
        self.time_mul = 1e0

    def call(self, q_cps, t_cps, data):
        _, q_dot_loss, q_ddot_loss, q_dddot_loss, torque_loss, q, q_dot, q_ddot, q_dddot, torque, t, t_cumsum, dt = super().call(q_cps, t_cps, data)
        centrifugal = None

        xyz = self.man.forward_kinematics(q)
        dx = (xyz[:, 1:, 0, 0] - xyz[:, :-1, 0, 0]) / dt[:, :-1]
        dy = (xyz[:, 1:, 1, 0] - xyz[:, :-1, 1, 0]) / dt[:, :-1]
        ddx = (dx[:, 1:] - dx[:, :-1]) / dt[:, :-2]
        ddy = (dy[:, 1:] - dy[:, :-1]) / dt[:, :-2]
        curvv = (dx[:, :-1] * ddy - ddx * dy[:, :-1]) / tf.sqrt(dx[:, :-1] ** 2 + dy[:, :-1] ** 2 + 1e-8)
        centrifugal_loss = tf.reduce_sum(tf.abs(curvv) * dt[:, 1:-1], axis=-1, keepdims=True)
        constraint_loss = self.end_effector_constraints_distance_function(xyz, dt)
        puck_pose = data[:, -2:]
        obstacle_loss = self.obstacle_distance_function(xyz, dt, puck_pose)[:, tf.newaxis]
        t_loss = huber(t[:, tf.newaxis])
        jerk_loss = tf.reduce_sum(tf.abs(q_dddot) * dt[..., tf.newaxis], axis=(1, 2))[:, tf.newaxis]
        int_torque_loss = tf.reduce_sum(tf.abs(torque) * dt[..., tf.newaxis], axis=(1, 2))[:, tf.newaxis]
        losses = tf.concat([tf.exp(self.alpha_q_dot) * q_dot_loss,
                            tf.exp(self.alpha_q_ddot) * q_ddot_loss,
                            tf.exp(self.alpha_torque) * torque_loss,
                            tf.exp(self.alpha_constraint) * constraint_loss,
                            tf.exp(self.alpha_obstacle) * obstacle_loss,
                            self.centrifugal_mul * centrifugal_loss,
                            self.time_mul * t_loss], axis=-1)
        unscaled_losses = tf.concat([q_dot_loss, q_ddot_loss, constraint_loss, centrifugal_loss, obstacle_loss, t_loss], axis=-1)
        sum_q_dot_loss = tf.reduce_sum(q_dot_loss, axis=-1)
        sum_q_ddot_loss = tf.reduce_sum(q_ddot_loss, axis=-1)
        sum_q_dddot_loss = tf.reduce_sum(q_dddot_loss, axis=-1)
        sum_constraint_loss = tf.reduce_sum(constraint_loss, axis=-1)
        sum_torque_loss = tf.reduce_sum(torque_loss, axis=-1)

        model_loss = tf.reduce_sum(losses, axis=-1)
        unscaled_model_loss = tf.reduce_sum(unscaled_losses, axis=-1)
        return model_loss, sum_constraint_loss, sum_q_dot_loss, sum_q_ddot_loss, sum_q_dddot_loss, sum_torque_loss, obstacle_loss, \
               q, q_dot, q_ddot, q_dddot, torque, centrifugal, xyz, t, t_cumsum, t_loss, dt, unscaled_model_loss, jerk_loss, int_torque_loss, centrifugal_loss

    def alpha_update(self, q_dot_loss, q_ddot_loss, q_dddot_loss, constraint_loss, torque_loss, obstacle_loss):
        max_alpha_update = 10.0
        alpha_q_dot_update = self.gamma * tf.clip_by_value(tf.math.log(q_dot_loss / self.bar_q_dot), -max_alpha_update, max_alpha_update)
        alpha_q_ddot_update = self.gamma * tf.clip_by_value(tf.math.log(q_ddot_loss / self.bar_q_ddot), -max_alpha_update, max_alpha_update)
        alpha_q_dddot_update = self.gamma * tf.clip_by_value(tf.math.log(q_dddot_loss / self.bar_q_dddot), -max_alpha_update, max_alpha_update)
        alpha_constraint_update = self.gamma * tf.clip_by_value(tf.math.log(constraint_loss / self.bar_constraint), -max_alpha_update, max_alpha_update)
        alpha_torque_update = self.gamma * tf.clip_by_value(tf.math.log(torque_loss / self.bar_torque), -max_alpha_update, max_alpha_update)
        alpha_obstacle_update = self.gamma * tf.clip_by_value(tf.math.log(obstacle_loss / self.bar_obstacle), -max_alpha_update, max_alpha_update)
        self.alpha_q_dot += alpha_q_dot_update
        self.alpha_q_ddot += alpha_q_ddot_update
        self.alpha_q_dddot += alpha_q_dddot_update
        self.alpha_constraint += alpha_constraint_update
        self.alpha_torque += alpha_torque_update
        self.alpha_obstacle += alpha_obstacle_update
