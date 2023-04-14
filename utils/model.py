from time import perf_counter
import numpy as np
import tensorflow as tf
from scipy.interpolate import BSpline as BSp
from scipy.interpolate import interp1d

from models.iiwa_obstacles_planner_boundaries import IiwaObstaclesPlannerBoundaries
from models.iiwa_planner_boundaries import IiwaPlannerBoundariesKinodynamic, IiwaPlannerBoundariesHitting


def load_model_hitting(path, N, bsp, bsp_t):
    model = IiwaPlannerBoundariesHitting(N, 3, 2, bsp, bsp_t)
    model(np.zeros([1, 38], dtype=np.float32))
    checkpoint = tf.train.Checkpoint(model=model)
    checkpoint.restore(path).expect_partial()
    return model


def load_model_kino(path, N, bsp, bsp_t):
    model = IiwaPlannerBoundariesKinodynamic(N, 3, 2, bsp, bsp_t)
    model(np.zeros([1, 20], dtype=np.float32))
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(path).expect_partial()
    return model


def load_model_obstacles(path, N, bsp, bsp_t):
    model = IiwaObstaclesPlannerBoundaries(N, 3, 3, bsp, bsp_t)
    model(np.zeros([1, 20], dtype=np.float32))
    ckpt = tf.train.Checkpoint(model=model)
    ckpt.restore(path).expect_partial()
    return model


def model_inference(model, data, bsp, bspt, expected_time=-1., uniform=False, freq=100):
    q_cps, t_cps = model(data)
    q = bsp.N @ q_cps
    q_dot_tau = bsp.dN @ q_cps
    q_ddot_tau = bsp.ddN @ q_cps

    dtau_dt = bspt.N @ t_cps
    ddtau_dtt = bspt.dN @ t_cps

    ts = 1. / dtau_dt[..., 0] / dtau_dt.shape[1]
    t = tf.cumsum(np.concatenate([np.zeros_like(ts[..., :1]), ts[..., :-1]], axis=-1), axis=-1)

    if expected_time > 0. and expected_time > t[0, -1]:
        ratio = t[0, -1] / expected_time
        t_cps *= ratio
        dtau_dt = bspt.N @ t_cps
        ddtau_dtt = bspt.dN @ t_cps

        ts = 1. / dtau_dt[..., 0] / dtau_dt.shape[1]
        t = tf.cumsum(np.concatenate([np.zeros_like(ts[..., :1]), ts[..., :-1]], axis=-1), axis=-1)

    q_dot = q_dot_tau * dtau_dt
    q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt

    if uniform:
        t = t[0]
        si = interp1d(t, np.linspace(0., 1., 1024), axis=-1)
        targ = np.linspace(t[0], t[-1], int(t[-1] * freq))
        s = si(targ)

        dtau_dt_bs = BSp(bspt.u, t_cps[0, :, 0], 7)
        ddtau_dtt_bs = dtau_dt_bs.derivative()
        q_bs = BSp(bsp.u, q_cps[0, :], 7)
        dq_bs = q_bs.derivative()
        ddq_bs = dq_bs.derivative()

        q = q_bs(s)
        q_dot_tau = dq_bs(s)
        q_ddot_tau = ddq_bs(s)
        dtau_dt = dtau_dt_bs(s)[..., np.newaxis]
        ddtau_dtt = ddtau_dtt_bs(s)[..., np.newaxis]
        t = targ

        q_dot = q_dot_tau * dtau_dt
        q_ddot = q_ddot_tau * dtau_dt ** 2 + ddtau_dtt * q_dot_tau * dtau_dt

        return q, q_dot, q_ddot, t
    return q.numpy()[0], q_dot.numpy()[0], q_ddot.numpy()[0], t.numpy()[0], q_cps.numpy()[0], t_cps.numpy()[0]
