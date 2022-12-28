import numpy as np
from scipy.interpolate import interp1d


def interpolate_trajectories(t, q, dq, ddq):
    t = np.concatenate([[0.], t[1:]], axis=0)
    ddqs = interp1d(t, ddq, axis=0)
    dqs = interp1d(t, dq, axis=0)
    qs = interp1d(t, q, axis=0)
    return t, qs, dqs, ddqs
