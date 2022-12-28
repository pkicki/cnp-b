import sys
import os
import inspect
import pinocchio as pino

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
sys.path.insert(0, os.path.dirname(parentdir))

import numpy as np
from utils.constants import TableConstraint, Limits, UrdfModels, Base, Cup, Robot, Table1, Table2
from utils.collisions import collision_with_box
from utils.manipulator import Iiwa

from utils.spo import StartPointOptimizer
from utils.spoo import StartPointOptimizerOrientation

urdf_path = os.path.join(os.path.dirname(__file__), "..", UrdfModels.iiwa_cup)
po = StartPointOptimizer(urdf_path)
poo = StartPointOptimizerOrientation(urdf_path)

pino_model = pino.buildModelFromUrdf(urdf_path)
pino_data = pino_model.createData()

man = Iiwa(urdf_path)


def validate_optimization(q, expected_xyz):
    pino.forwardKinematics(pino_model, pino_data, np.pad(q, (0, 7 - len(q)), mode='constant'))
    pino.updateFramePlacements(pino_model, pino_data)
    xyz = pino_data.oMf[-1].translation
    o = pino_data.oMf[-1].rotation
    xyz_error = np.linalg.norm(xyz - expected_xyz)
    return xyz_error < 1e-2


def validate_torques(q):
    tau = pino.rnea(pino_model, pino_data, q, np.zeros_like(q), np.zeros_like(q))
    error = np.abs(tau) - Limits.tau7
    valid = np.all(error < 0)
    return valid


if __name__ == "__main__":
    data = []
    ds = sys.argv[1]
    assert ds in ["train", "val", "test", "dummy"]
    idx = int(sys.argv[2])
    N = int(sys.argv[3])

    x0l = Table1.xl
    x0h = Table1.xh
    y0l = Table1.yl
    y0h = Table1.yh
    z0l = Table1.z_range_l
    z0h = Table1.z_range_h

    xkl = Table2.xl
    xkh = Table2.xh
    ykl = Table2.yl
    ykh = Table2.yh
    zkl = Table2.z_range_l
    zkh = Table2.z_range_h

    height = Cup.height
    radius = Robot.radius

    i = 0
    while i < N:
        def draw(l, h, n=1):
            return (h - l) * np.random.rand(n) + l


        x0 = draw(x0l, x0h)
        y0 = draw(y0l, y0h)
        z0 = draw(z0l, z0h)

        xk = draw(xkl, xkh)
        yk = draw(ykl, ykh)
        zk = draw(zkl, zkh)

        qlow = np.array([-np.pi / 2, 0.0, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi])
        qhigh = np.array([np.pi / 2, np.pi / 2, np.pi / 2, 0.0, np.pi / 2, np.pi / 2, np.pi])

        q0_init = draw(qlow, qhigh, n=7)
        xyz = np.concatenate([x0, y0, z0])
        # q0 = po.solve(xyz, q0_init)
        # q0_init = po.solve(xyz, q0_init)[:6]
        q0 = poo.solve(xyz, q0_init)
        if not validate_optimization(q0, xyz):
            continue
        links_poses = man.interpolated_forward_kinematics(q0)[0][..., 0]
        o = np.ones_like(z0)
        collision = collision_with_box(links_poses, radius, x0l * o, x0h * o, y0l * o, y0h * o, -1e10 * o, z0 - height)
        if np.sum(collision): continue
        collision = collision_with_box(links_poses, radius, xkl * o, xkh * o, ykl * o, ykh * o, -1e10 * o, zk - height)
        if np.sum(collision): continue

        #qk_init = draw(qlow, qhigh, n=7)
        xyz = np.concatenate([xk, yk, zk])
        # qk = po.solve([xk, yk, zk], qk_init)
        # qk = po.solve([xk, yk, zk])
        # qk_init = po.solve(xyz, qk_init)[:6]
        qk = poo.solve(xyz, q0)
        if not validate_optimization(qk, xyz):
            continue
        links_poses = man.interpolated_forward_kinematics(qk)[0][..., 0]
        collision = collision_with_box(links_poses, radius, x0l * o, x0h * o, y0l * o, y0h * o, -1e10 * o, z0 - height)
        if np.sum(collision): continue
        collision = collision_with_box(links_poses, radius, xkl * o, xkh * o, ykl * o, ykh * o, -1e10 * o, zk - height)
        if np.sum(collision): continue

        if not validate_torques(q0): continue
        if not validate_torques(qk): continue

        data.append(q0.tolist() + qk.tolist() + [x0, y0, z0] + [xk, yk, zk])
        i += 1

    # dir_name = f"paper/airhockey_table_moves_v08_a10v_tilted_93/{ds}"
    dir_name = f"paper/kinodynamic7fixed_12kg_validated/{ds}"
    os.makedirs(dir_name, exist_ok=True)
    np.savetxt(f"{dir_name}/data_{N}_{idx}.tsv", data, delimiter='\t', fmt="%.8f")
    os.popen(f'cp {os.path.basename(__file__)} {dir_name}')
