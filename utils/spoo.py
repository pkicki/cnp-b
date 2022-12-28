from time import time

import numpy as np
from scipy import optimize as spo
import pinocchio as pino

from utils.constants import UrdfModels, TableConstraint, Base
from utils.manipulator import Iiwa


class StartPointOptimizerOrientation:
    def __init__(self, urdf_path, n=7):
        self.n = n
        self.model = pino.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.bounds = spo.Bounds(self.model.lowerPositionLimit[:n], self.model.upperPositionLimit[:n])

    def solve(self, point, q0=None):
        point = point.flatten()
        if q0 is None:
            q0 = Base.configuration
        options = {'maxiter': 300, 'ftol': 1e-06, 'iprint': 1, 'disp': False,
                   'eps': 1.4901161193847656e-08, 'finite_diff_rel_step': None}
        t0 = time()
        r = spo.minimize(lambda x: self.f(x, point), q0, method='SLSQP',
                         bounds=self.bounds, options=options)
        t1 = time()
        print(r)
        print("TIME:", t1 - t0)
        return r.x

    def f(self, q, hit_point):
        pino.forwardKinematics(self.model, self.data, q)
        pino.updateFramePlacements(self.model, self.data)
        x = self.data.oMf[-1].translation
        diff_x = x - hit_point
        o = self.data.oMf[-1].rotation
        diff_dir = 1.0 - o[2, 2]
        return np.linalg.norm(diff_x) + np.linalg.norm(diff_dir)


if __name__ == "__main__":
    urdf_path = "../" + UrdfModels.iiwa_cup
    #point = np.array([0.4, 0.4, 0.4])
    #point = np.array([0.4, 0.4, 0.3])
    point = np.array([0.237, -0.38, 0.32])
    #po = StartPointOptimizer(urdf_path, 7)
    po = StartPointOptimizerOrientation(urdf_path)

    qlow = np.array([-np.pi / 2, 0.0, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi / 2, -np.pi])
    qhigh = np.array([np.pi / 2, np.pi / 2, np.pi / 2, 0.0, np.pi / 2, np.pi / 2, np.pi])
    for i in range(100):
        qinit = (qhigh - qlow) * np.random.rand(7) + qlow
        q = po.solve(point, qinit)
        pino.forwardKinematics(po.model, po.data, q)
        pino.updateFramePlacements(po.model, po.data)
        xyz = po.data.oMf[-1].translation
        o = po.data.oMf[-1].rotation
        xyz_error = np.linalg.norm(xyz - point)
        a = 0
