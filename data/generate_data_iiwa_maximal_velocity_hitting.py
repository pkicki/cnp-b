import sys
import os
import inspect
import pinocchio as pino


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, os.path.dirname(parentdir))
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
from utils.constants import TableConstraint, Limits, UrdfModels, Base
from utils.spo import StartPointOptimizer
from utils.velocity_projection import VelocityProjector
from utils.hpo_interface import get_hitting_configuration_opt
from data.generate_data_iiwa_universal_planner import get_hitting_configuration
from data.generate_data_iiwa_hitting import validate_if_initial_mallet_and_puck_positions_makes_hit_possible, \
    validate_if_pose_is_reachable_with_given_velocity

if __name__ == "__main__":
    urdf_path = os.path.join(os.path.dirname(__file__), "..", UrdfModels.striker)
    po = StartPointOptimizer(urdf_path)
    vp = VelocityProjector(urdf_path)

    pino_model = pino.buildModelFromUrdf(urdf_path)
    pino_data = pino_model.createData()

    data = []
    ds = sys.argv[1]
    assert ds in ["train", "val", "test", "dummy"]
    idx = int(sys.argv[2])
    N = int(sys.argv[3])

    x0l = 0.60
    x0h = 0.7
    y0l = -0.05
    y0h = 0.05
    xkl = 0.65
    xkh = 1.3
    ykl = -0.45
    ykh = 0.45
    i = 0
    while i < N:
        x0 = (x0h - x0l) * np.random.rand() + x0l
        y0 = (y0h - y0l) * np.random.rand() + y0l

        xk = (xkh - xkl) * np.random.rand() + xkl
        yk = (ykh - ykl) * np.random.rand() + ykl

        if not validate_if_initial_mallet_and_puck_positions_makes_hit_possible(x0, y0, xk, yk):
            continue


        point = np.array([x0, y0, TableConstraint.Z])
        q0 = po.solve(point)
        q_dot_0 = np.zeros((6))
        q_ddot_0 = np.zeros((6))

        xg = 2.49
        yg = 0.
        # Direction 1: straight
        thk = np.arctan2(yg - yk, xg - xk)
        qk, q_dot_k, vk = get_hitting_configuration(xk, yk, thk, q0.tolist())
        if qk is None or q_dot_k is None:
            continue
        q_dot_k = np.array(q_dot_k)
        ql = Limits.q_dot
        max_gain = np.min(ql / np.abs(q_dot_k[:6]))
        q_dot_k = max_gain * q_dot_k

        if not validate_if_pose_is_reachable_with_given_velocity(xk, yk, vk[0] * max_gain, vk[1] * max_gain):
            continue
        if not validate_if_pose_is_reachable_with_given_velocity(xk, yk, -vk[0] * max_gain, -vk[1] * max_gain):
            continue


        data.append(q0.tolist() + qk + [xk, yk, thk] + q_dot_0.tolist() + [0.] + q_ddot_0.tolist() + [0.] + q_dot_k.tolist())
        i += 1

    dir_name = f"paper/airhockey_maximal_velocity_hitting_new/{ds}"
    ranges = [x0l, x0h, y0l, y0h, xkl, xkh, ykl, ykh]
    os.makedirs(dir_name, exist_ok=True)
    np.savetxt(f"{dir_name}/data_{N}_{idx}.tsv", data, delimiter='\t', fmt="%.8f")
    os.popen(f'cp {os.path.basename(__file__)} {dir_name}')