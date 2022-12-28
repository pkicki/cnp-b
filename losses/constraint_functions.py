import tensorflow as tf
import numpy as np

from losses.utils import huber
from utils.collisions import collision_with_box, simple_collision_with_box
from utils.constants import Table1, Cup, Robot, Table2
from utils.data import unpack_data_kinodynamic
from utils.table import Table

table = Table()


def air_hockey_table(xyz, dt):
    xyz = xyz[..., 0]
    huber_along_path = lambda x: tf.reduce_sum(dt * huber(x), axis=-1)  # / tf.reduce_sum(dt, axis=-1)
    relu_huber_along_path = lambda x: huber_along_path(tf.nn.relu(x))
    xlow_loss = relu_huber_along_path(table.xlb - xyz[..., 0])
    xhigh_loss = relu_huber_along_path(xyz[..., 0] - table.xrt)
    ylow_loss = relu_huber_along_path(table.ylb - xyz[..., 1])
    yhigh_loss = relu_huber_along_path(xyz[..., 1] - table.yrt)
    z_loss = huber_along_path(xyz[..., 2] - table.z)
    constraint_losses = tf.stack([xlow_loss, xhigh_loss, ylow_loss, yhigh_loss, z_loss], axis=-1)
    return constraint_losses


def air_hockey_puck(xyz, dt, puck_pose):
    xy = xyz[:, :, :2, 0]
    dist_from_puck = tf.sqrt(tf.reduce_sum((puck_pose[:, tf.newaxis] - xy) ** 2, axis=-1))
    puck_loss = tf.nn.relu(0.09 - dist_from_puck)
    idx_ = tf.argmin(puck_loss[..., ::-1], axis=-1)
    idx = tf.cast(xyz.shape[1] - idx_, tf.float32)[:, tf.newaxis] - 1
    range = tf.range(xyz.shape[1], dtype=tf.float32)[tf.newaxis]
    threshold = tf.where(idx > range, tf.ones_like(puck_loss), tf.zeros_like(puck_loss))

    puck_loss = tf.reduce_sum(puck_loss * threshold * dt, axis=-1)
    return puck_loss


def two_tables_vertical(xyz, R, dt, data):
    huber_along_path = lambda x: tf.reduce_sum(dt * huber(x), axis=-1)

    first_z = xyz[:, :1, -1:, -1].numpy()
    last_z = xyz[:, -1:, -1:, -1].numpy()
    o = np.ones_like(last_z)
    collision_table_1 = collision_with_box(xyz, Robot.radius, Table1.xl * o, Table1.xh * o,
                                           Table1.yl * o, Table1.yh * o, -1e10 * o,
                                           first_z - Cup.height)
    collision_table_2 = collision_with_box(xyz, Robot.radius, Table2.xl * o, Table2.xh * o,
                                           Table2.yl * o, Table2.yh * o, -1e10 * o,
                                           last_z - Cup.height)

    vertical_loss = huber_along_path(1.0 - R[:, :, 2, 2])
    collision_table_1_loss = huber_along_path(tf.reduce_sum(collision_table_1, axis=-1))
    collision_table_2_loss = huber_along_path(tf.reduce_sum(collision_table_2, axis=-1))
    constraint_losses = tf.stack([vertical_loss, collision_table_1_loss, collision_table_2_loss], axis=-1)
    return constraint_losses


def two_tables_vertical_objectcollision(xyz, R, dt, data):
    two_tables_vertical_loss = two_tables_vertical(xyz, R, dt, data)
    xyz_end = xyz[:, :, -1:]
    h = Cup.height
    w = Cup.width
    xyz_cuboid = np.array([[w, w, h], [w, w, -h], [w, -w, h], [w, -w, -h],
                           [-w, w, h], [-w, w, -h], [-w, -w, h], [-w, -w, -h]
                           ])[np.newaxis, np.newaxis]
    xyz_corners = xyz_end + (R[:, :, tf.newaxis] @ xyz_cuboid[..., np.newaxis])[..., 0]
    first_z = xyz[:, :1, -1:, -1].numpy()
    last_z = xyz[:, -1:, -1:, -1].numpy()
    o = np.ones_like(last_z)
    collision_table_1 = simple_collision_with_box(xyz_corners, Table1.xl * o, Table1.xh * o,
                                                  Table1.yl * o, Table1.yh * o, -1e10 * o,
                                                  first_z - Cup.height)
    collision_table_2 = simple_collision_with_box(xyz_corners, Table2.xl * o, Table2.xh * o,
                                                  Table2.yl * o, Table2.yh * o, -1e10 * o,
                                                  last_z - Cup.height)
    huber_along_path = lambda x: tf.reduce_sum(dt * huber(x), axis=-1)
    collision_table_1_loss = huber_along_path(tf.reduce_sum(collision_table_1, axis=-1))
    collision_table_2_loss = huber_along_path(tf.reduce_sum(collision_table_2, axis=-1))
    constraint_losses = tf.concat([two_tables_vertical_loss, collision_table_1_loss[..., tf.newaxis],
                                   collision_table_2_loss[..., tf.newaxis]], axis=-1)
    return constraint_losses


def two_tables_object_collision(xyz, R, dt, data):
    huber_along_path = lambda x: tf.reduce_sum(dt * huber(x), axis=-1)

    first_z = xyz[:, :1, -1:, -1].numpy()
    last_z = xyz[:, -1:, -1:, -1].numpy()
    o = np.ones_like(last_z)
    robot_collision_table_1 = collision_with_box(xyz, Robot.radius, Table1.xl * o, Table1.xh * o,
                                                 Table1.yl * o, Table1.yh * o, -1e10 * o,
                                                 first_z - Cup.height)
    robot_collision_table_2 = collision_with_box(xyz, Robot.radius, Table2.xl * o, Table2.xh * o,
                                                 Table2.yl * o, Table2.yh * o, -1e10 * o,
                                                 last_z - Cup.height)

    robot_collision_table_1_loss = huber_along_path(tf.reduce_sum(robot_collision_table_1, axis=-1))
    robot_collision_table_2_loss = huber_along_path(tf.reduce_sum(robot_collision_table_2, axis=-1))

    xyz_end = xyz[:, :, -1:]
    h = Cup.height
    w = Cup.width
    xyz_cuboid = np.array([
                            # corners
                           [w, w, h], [w, w, -h], [w, -w, h], [w, -w, -h],
                           [-w, w, h], [-w, w, -h], [-w, -w, h], [-w, -w, -h],
                           # middle points on the edges
                           [w, w, 0], [w, -w, 0], [-w, w, 0], [-w, -w, 0],
                           [w, 0, h], [w, 0, -h], [-w, 0, h], [-w, 0, -h],
                           [0, w, h], [0, w, -h], [0, -w, h], [0, -w, -h],
                           # middle points on the faces
                           [w, 0, 0], [-w, 0, 0],
                           [0, w, 0], [0, -w, 0],
                           [0, 0, h], [0, 0, -h],
                           ])[np.newaxis, np.newaxis]
    xyz_object = xyz_end + (R[:, :, tf.newaxis] @ xyz_cuboid[..., np.newaxis])[..., 0]

    object_collision_table_1 = simple_collision_with_box(xyz_object, Table1.xl * o, Table1.xh * o,
                                                         Table1.yl * o, Table1.yh * o, -1e10 * o,
                                                         first_z - Cup.height)
    object_collision_table_2 = simple_collision_with_box(xyz_object, Table2.xl * o, Table2.xh * o,
                                                         Table2.yl * o, Table2.yh * o, -1e10 * o,
                                                         last_z - Cup.height)
    huber_along_path = lambda x: tf.reduce_sum(dt * huber(x), axis=-1)
    object_collision_table_1_loss = huber_along_path(tf.reduce_sum(object_collision_table_1, axis=-1))
    object_collision_table_2_loss = huber_along_path(tf.reduce_sum(object_collision_table_2, axis=-1))
    constraint_losses = tf.stack([robot_collision_table_1_loss, robot_collision_table_2_loss,
                                  object_collision_table_1_loss, object_collision_table_2_loss,
                                  ], axis=-1)
    return constraint_losses


def two_tables_vertical_end(xyz, R, dt, data):
    two_tables_vertical_loss = two_tables_vertical(xyz, R, dt, data)
    q0, qd, xyz0, xyzk, q_dot_0, q_dot_d, q_ddot_0 = unpack_data_kinodynamic(data, 7)
    xyz_end = xyz[..., -1, -1, :]
    relu_huber = lambda x: huber(tf.nn.relu(x))
    border = 0.05
    xl = Table2.xl + border
    xh = Table2.xh - border
    yl = Table2.yl + border
    yh = Table2.yh - border
    xlow_loss = relu_huber(xl - xyz_end[..., 0])
    xhigh_loss = relu_huber(xyz_end[..., 0] - xh)
    ylow_loss = relu_huber(yl - xyz_end[..., 1])
    yhigh_loss = relu_huber(xyz_end[..., 1] - yh)
    z_loss = huber(xyz_end[..., 2] - xyzk[..., 2])
    xyz_losses = tf.stack([xlow_loss, xhigh_loss, ylow_loss, yhigh_loss, z_loss], axis=-1)
    constraint_losses = tf.concat([two_tables_vertical_loss, xyz_losses], axis=-1)
    return constraint_losses
