import numpy as np

xlb = 0.55
ylb = -0.5
xrt = 2.
yrt = 0.5
z = 0.16
relu = lambda x: np.maximum(x, 0.)


def compute_cartesian_losses(ee):
    xlow_loss = relu(xlb - ee[..., 0])
    xhigh_loss = relu(ee[..., 0] - xrt)
    ylow_loss = relu(ylb - ee[..., 1])
    yhigh_loss = relu(ee[..., 1] - yrt)
    z_loss = np.abs(ee[..., 2] - z)
    return z_loss, xlow_loss, xhigh_loss, ylow_loss, yhigh_loss


def check_if_plan_valid(ee, q, q_dot, q_ddot, torque):
    z_loss, xlow_loss, xhigh_loss, ylow_loss, yhigh_loss = compute_cartesian_losses(ee)

    torque_limits = 0.9 * np.array([320, 320, 176, 176, 110, 40], dtype=np.float32)[np.newaxis]
    q_dot_limits = 1.0 * np.array([1.4835, 1.4835, 1.7453, 1.3090, 2.2689, 2.3562], dtype=np.float32)[np.newaxis]
    q_ddot_limits = 10. * q_dot_limits
    q_dot_loss = relu(np.abs(q_dot[:, :6]) - q_dot_limits)
    q_ddot_loss = relu(np.abs(q_ddot[:, :6]) - q_ddot_limits)
    torque_loss = relu(np.abs(torque[:, :6]) - torque_limits)

    valid_z = np.all(z_loss < 0.01)
    valid_xl = np.all(xlow_loss == 0.0)
    valid_xh = np.all(xhigh_loss == 0.0)
    valid_yl = np.all(ylow_loss == 0.0)
    valid_yh = np.all(yhigh_loss == 0.0)
    valid_q_dot = np.all(q_dot_loss == 0.0)
    valid_q_ddot = np.all(q_ddot_loss == 0.0)
    valid_torque = np.all(torque_loss == 0.0)

    valids = [valid_z, valid_xl, valid_xh, valid_yl, valid_yh, valid_q_dot, valid_q_ddot, valid_torque]
    return np.all(valids)
