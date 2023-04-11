import sys
import os
import inspect
import pinocchio as pino


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, os.path.dirname(parentdir))
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import networkx as nx
#import tensorflow as tf
from utils.constants import TableConstraint, Limits, UrdfModels, Base

#from models.iiwa_ik_hitting import IiwaIKHitting
#from utils.execution import ExperimentHandler

from utils.spo import StartPointOptimizer
from utils.velocity_projection import VelocityProjector
from utils.hpo_interface import get_hitting_configuration_opt

def check_connection(G, idxs1, idxs2):
    separated = False
    for idx1 in idxs1:
        for idx2 in idxs2:
            separated = nx.has_path(G, idx1, idx2)
            if separated: break
        if separated: break
    return separated

if __name__ == "__main__":
    data = []
    ds = sys.argv[1]
    assert ds in ["train", "val", "test", "dummy"]
    idx = int(sys.argv[2])
    N = int(sys.argv[3])

    x0l = 0.0
    x0h = 0.1
    y0l = 0.0
    y0h = 0.1
    xkl = 0.9
    xkh = 1.0
    ykl = 0.9
    ykh = 1.0
    i = 0
    while i < N:
        x0 = (x0h - x0l) * np.random.rand() + x0l
        y0 = (y0h - y0l) * np.random.rand() + y0l

        xk = (xkh - xkl) * np.random.rand() + xkl
        yk = (ykh - ykl) * np.random.rand() + ykl

        n_obs = 10
        obs_xy = np.random.random((n_obs, 2))
        obs_r = 0.4 * np.random.random((n_obs, 1)) + 0.1
        obs_r = obs_r * (np.random.random((n_obs, 1)) < 0.5).astype(np.float32)

        if_left_wall_connected = obs_xy[..., 0] < obs_r[:, 0]
        if_right_wall_connected = obs_xy[..., 0] > 1. - obs_r[:, 0]
        if_top_wall_connected = obs_xy[..., 1] > 1. - obs_r[:, 0]
        if_bottom_wall_connected = obs_xy[..., 1] < obs_r[:, 0]

        if_obs_connected = np.linalg.norm(obs_xy[:, np.newaxis] - obs_xy[np.newaxis], axis=-1) < obs_r[:, np.newaxis, 0] + obs_r[np.newaxis, :, 0]
        G = nx.from_numpy_matrix(if_obs_connected)

        bottom_idxs = np.where(if_bottom_wall_connected)[0]
        top_idxs = np.where(if_top_wall_connected)[0]
        left_idxs = np.where(if_left_wall_connected)[0]
        right_idxs = np.where(if_right_wall_connected)[0]

        if check_connection(G, bottom_idxs, top_idxs): continue
        if check_connection(G, bottom_idxs, left_idxs): continue
        if check_connection(G, left_idxs, right_idxs): continue
        if check_connection(G, top_idxs, right_idxs): continue


        obstacles = np.concatenate([obs_xy, obs_r], axis=-1)
        obstacles = obstacles.reshape(-1)

        xy0 = np.array([x0, y0])[np.newaxis]
        xyk = np.array([xk, yk])[np.newaxis]
        xy0_dists = np.linalg.norm(xy0 - obs_xy, axis=-1)
        xyk_dists = np.linalg.norm(xyk - obs_xy, axis=-1)
        xy0_invalid = np.any(xy0_dists < obs_r)
        xyk_invalid = np.any(xyk_dists < obs_r)
        if xy0_invalid or xyk_invalid:
            continue

        data.append([x0, y0, 0., 0., xk, yk, 0., 0.] + obstacles.tolist())
        i += 1

    # dir_name = f"paper/airhockey_table_moves_v08_a10v_tilted_93/{ds}"
    dir_name = f"paper/obstacles2D_boundaries/{ds}"
    os.makedirs(dir_name, exist_ok=True)
    np.savetxt(f"{dir_name}/data_{N}_{idx}.tsv", data, delimiter='\t', fmt="%.8f")
    os.popen(f'cp {os.path.basename(__file__)} {dir_name}')