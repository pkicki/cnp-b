import numpy as np


def bound_data(data):
    M = 14
    q = data[..., :M]
    q = np.arctan2(np.sin(q), np.cos(q))
    return np.concatenate([q, data[..., M:]], axis=-1)


def unpack_data_boundaries(x, n):
    q0 = x[:, :n - 1]
    qk = x[:, n:2 * n - 1]
    xyth = x[:, 2 * n: 2 * n + 3]
    q_dot_0 = x[:, 2 * n + 3: 3 * n + 2]
    q_ddot_0 = x[:, 3 * n + 3: 4 * n + 2]
    q_dot_k = x[:, 4 * n + 3: 5 * n + 2]
    puck_pose = x[:, -2:]
    return q0, qk, xyth, q_dot_0, q_dot_k, q_ddot_0, puck_pose


def unpack_data_boundaries_heights(x, n):
    q0 = x[:, :n - 1]
    qk = x[:, n:2 * n - 1]
    xyth = x[:, 2 * n: 2 * n + 3]
    q_dot_0 = x[:, 2 * n + 3: 3 * n + 2]
    q_ddot_0 = x[:, 3 * n + 3: 4 * n + 2]
    q_dot_k = x[:, 4 * n + 3: 5 * n + 2]
    table_height = x[:, -1:]
    return q0, qk, xyth, q_dot_0, q_dot_k, q_ddot_0, table_height


def unpack_data_kinodynamic(x, n):
    q0 = x[:, :n]
    qk = x[:, n:2 * n]
    xyz0 = x[:, 2 * n: 2 * n + 3]
    xyzk = x[:, 2 * n + 3: 2 * n + 6]
    q_dot_0 = np.zeros_like(q0)
    q_ddot_0 = np.zeros_like(q0)
    q_dot_k = np.zeros_like(qk)
    return q0, qk, xyz0, xyzk, q_dot_0, q_dot_k, q_ddot_0


def unpack_data_acrobot(x):
    q0 = x[:, :2]
    qk = x[:, 4:6]
    q_dot_0 = x[:, 2:4]
    q_ddot_0 = np.zeros_like(q0)
    q_dot_k = x[:, 6:8]
    return q0, qk, None, None, q_dot_0, q_dot_k, q_ddot_0


def unpack_data_linear_move(x, n):
    q0 = x[:, :n - 1]
    xyz0 = x[:, n:n + 3]
    xyzk = x[:, n + 3:n + 6]
    q_dot_0 = x[:, n + 6: 2 * n + 5]
    q_ddot_0 = x[:, 2 * n + 6: 3 * n + 5]
    return q0, xyz0, xyzk, q_dot_0, q_ddot_0


def unpack_data_obstacles2D(x):
    xy0 = x[:, :2]
    dxy0 = x[:, 2:4]
    xyk = x[:, 4:6]
    dxyk = x[:, 6:8]
    obstacles = x[:, 8:]
    return xy0, xyk, dxy0, dxyk, obstacles


def unpack_data_iiwa_obstacles(x):
    n = 7
    q0 = x[:, :n]
    qk = x[:, n:2 * n]
    obstacles = x[:, 2 * n:]
    q_dot_0 = np.zeros_like(q0)
    q_ddot_0 = np.zeros_like(q0)
    q_dot_k = np.zeros_like(qk)
    q_ddot_k = np.zeros_like(q0)
    return q0, qk, q_dot_0, q_dot_k, q_ddot_0, q_ddot_k, obstacles


