import numpy as np


class TableConstraint:
    XLB = 0.6
    YLB = -0.45
    XRT = 2.
    YRT = 0.45
    Z = 0.16

    @staticmethod
    def in_table_xy(x, y):
        return TableConstraint.XLB <= x <= TableConstraint.XRT and TableConstraint.YLB <= y <= TableConstraint.YRT


class Table1:
    xl = 0.2
    xh = 0.6
    yl = -0.6
    yh = -0.3
    z_range_l = 0.2
    z_range_h = 0.5


class Table2:
    xl = 0.2
    xh = 0.6
    yl = 0.3
    yh = 0.6
    z_range_l = 0.2
    z_range_h = 0.5


class Cup:
    height = 0.15
    width = 0.1


class Robot:
    radius = 0.12


class Limits:
    q7 = np.array([2.967, 2.094, 2.967, 2.094, 2.967, 2.094, 3.054])
    q = q7[:6]
    q_dot7 = 0.8 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562, 2.3562], dtype=np.float32)
    q_dot = q_dot7[:6]
    q_ddot7 = 10. * q_dot7
    q_ddot7 = np.min(np.stack([q_ddot7, 20. * np.ones((7,), dtype=np.float32)], axis=-1), axis=-1)
    q_ddot = q_ddot7[:6]
    tau7 = 0.8 * np.array([320, 320, 176, 176, 110, 40, 40], dtype=np.float32)
    tau = tau7[:6]
    q_dddot7 = 5 * q_ddot7
    q_dddot = q_dddot7[:6]


class Base:
    configuration = [-7.16000830e-06, 6.97494070e-01, 7.26955352e-06, -5.04898567e-01, 6.60813111e-07, 1.92857916e+00]
    configuration7 = [-7.16000830e-06, 6.97494070e-01, 7.26955352e-06, -5.04898567e-01, 6.60813111e-07, 1.92857916e+00, 0.]
    position = [0.65, 0., 0.16]


class UrdfModels:
    striker = "iiwa_striker.urdf"
    iiwa = "iiwa.urdf"
    iiwa_cup = "iiwa_cup.urdf"


class Env:
    n_obs = 3
