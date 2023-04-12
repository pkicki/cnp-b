import os
import tensorflow as tf

from losses.feasibility import FeasibilityLoss
from losses.feasibility2D import Feasibility2DLoss
from losses.utils import huber
from utils.bspline import BSpline
from utils.collisions import collision_with_circle, collision_with_boundary
from utils.constants import UrdfModels
from utils.data import unpack_data_obstacles2D
from utils.manipulator import Iiwa


class Obstacles2DLoss(Feasibility2DLoss):
    def __init__(self, N):
        super(Obstacles2DLoss, self).__init__(N)
        self.alpha_obstacle = tf.math.log(1e-0)
        #self.alpha_xy_dot = tf.math.log(1e-2)
        #self.alpha_xy_ddot = tf.math.log(1e-4)
        self.alpha_xy_dot = tf.math.log(1e-0)
        self.alpha_xy_ddot = tf.math.log(1e-0)
        self.gamma = 1e-2
        self.bar_obstacle = 1e-9
        self.bar_xy_dot = 6e-3
        self.bar_xy_ddot = 6e-2
        self.time_mul = 1e0

    def call(self, xy_cps, t_cps, data):
        _, xy_dot_loss, xy_ddot_loss, xy, xy_dot, xy_ddot, t, t_cumsum, dt = super().call(xy_cps, t_cps, data)

        xy_dot_loss = xy_dot_loss[:, tf.newaxis]
        xy_ddot_loss = xy_ddot_loss[:, tf.newaxis]

        xy0, xyk, dxy0, dxyk, obstacles = unpack_data_obstacles2D(data)
        obs = tf.reshape(obstacles, (-1, 10, 3))
        obs_xy = obs[..., :2]
        obs_r = obs[..., -1]
        obstacle_loss = collision_with_circle(xy, obs_xy, obs_r)
        obstacle_loss = huber(obstacle_loss)
        obstacle_loss = tf.reduce_sum(obstacle_loss * dt[..., tf.newaxis], axis=1)
        boundary_loss = collision_with_boundary(xy)
        boundary_loss = huber(boundary_loss)
        boundary_loss = tf.reduce_sum(boundary_loss * dt, axis=1)[..., tf.newaxis]
        obstacle_loss = tf.concat([obstacle_loss, boundary_loss], axis=-1)

        t_loss = huber(t[:, tf.newaxis])
        losses = tf.concat([tf.exp(self.alpha_xy_dot) * xy_dot_loss,
                            tf.exp(self.alpha_xy_ddot) * xy_ddot_loss,
                            tf.exp(self.alpha_obstacle) * obstacle_loss,
                            self.time_mul * t_loss], axis=-1)
        unscaled_losses = tf.concat([xy_dot_loss, xy_ddot_loss, obstacle_loss, t_loss], axis=-1)
        sum_xy_dot_loss = tf.reduce_sum(xy_dot_loss, axis=-1)
        sum_xy_ddot_loss = tf.reduce_sum(xy_ddot_loss, axis=-1)

        model_loss = tf.reduce_sum(losses, axis=-1)
        unscaled_model_loss = tf.reduce_sum(unscaled_losses, axis=-1)
        return model_loss, sum_xy_dot_loss, sum_xy_ddot_loss, obstacle_loss, \
               xy, xy_dot, xy_ddot, t, t_cumsum, t_loss, dt, unscaled_model_loss

    def alpha_update(self, xy_dot_loss, xy_ddot_loss, obstacle_loss):
        max_alpha_update = 10.0
        alpha_xy_dot_update = self.gamma * tf.clip_by_value(tf.math.log(xy_dot_loss / self.bar_xy_dot), -max_alpha_update, max_alpha_update)
        alpha_xy_ddot_update = self.gamma * tf.clip_by_value(tf.math.log(xy_ddot_loss / self.bar_xy_ddot), -max_alpha_update, max_alpha_update)
        alpha_obstacle_update = self.gamma * tf.clip_by_value(tf.math.log(obstacle_loss / self.bar_obstacle), -max_alpha_update, max_alpha_update)
        self.alpha_xy_dot += alpha_xy_dot_update
        self.alpha_xy_ddot += alpha_xy_ddot_update
        self.alpha_obstacle += alpha_obstacle_update


class Obstacles2DPathLoss:
    def __init__(self, N):
        self.alpha_obstacle = tf.math.log(1e-0)
        #self.alpha_xy_dot = tf.math.log(1e-2)
        #self.alpha_xy_ddot = tf.math.log(1e-4)
        self.gamma = 1e-2
        #self.bar_obstacle = 1e-9
        self.bar_obstacle = 1e-5
        self.bsp = BSpline(N)

    def call(self, xy_cps, data):
        xy = self.bsp.N @ xy_cps

        ds = tf.sqrt(tf.reduce_sum(tf.square(xy[:, 1:] - xy[:, :-1]), axis=-1))
        xy0, xyk, dxy0, dxyk, obstacles = unpack_data_obstacles2D(data)
        obs = tf.reshape(obstacles, (-1, 10, 3))
        obs_xy = obs[..., :2]
        obs_r = obs[..., -1]
        obstacle_loss = collision_with_circle(xy, obs_xy, obs_r)
        obstacle_loss = huber(obstacle_loss)
        obstacle_loss = tf.reduce_sum(obstacle_loss[:, 1:] * ds[..., tf.newaxis], axis=1)
        boundary_loss = collision_with_boundary(xy)
        boundary_loss = huber(boundary_loss)
        boundary_loss = tf.reduce_sum(boundary_loss[:, 1:] * ds, axis=1)[..., tf.newaxis]
        obstacle_loss = tf.concat([obstacle_loss, boundary_loss], axis=-1)
        
        s_loss = huber(tf.reduce_sum(ds, axis=-1))[:, tf.newaxis]

        losses = tf.concat([
            tf.exp(self.alpha_obstacle) * obstacle_loss,
            s_loss], axis=-1)
        unscaled_losses = tf.concat([obstacle_loss, s_loss], axis=-1)

        model_loss = tf.reduce_sum(losses, axis=-1)
        unscaled_model_loss = tf.reduce_sum(unscaled_losses, axis=-1)
        return model_loss, s_loss, obstacle_loss, xy, unscaled_model_loss

    def alpha_update(self, obstacle_loss):
        max_alpha_update = 10.0
        alpha_obstacle_update = self.gamma * tf.clip_by_value(tf.math.log(obstacle_loss / self.bar_obstacle), -max_alpha_update, max_alpha_update)
        self.alpha_obstacle += alpha_obstacle_update

    def __call__(self, xy_cps, data):
        return self.call(xy_cps, data)
