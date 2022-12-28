import numpy as np
import scipy
from scipy import optimize as spo
import pinocchio as pino

from utils.constants import UrdfModels, Limits


class VelocityProjector:
    def __init__(self, urdf_path, n=9):
        self.n = n
        self.model = pino.buildModelFromUrdf(urdf_path)
        self.data = self.model.createData()
        self.bounds = spo.Bounds(self.model.lowerPositionLimit[:7], self.model.upperPositionLimit[:7])

    def compute_q_dot(self, q, v_xyz_scale):
        q = np.pad(q, (0, self.n - q.shape[0]), mode='constant')
        idx_ = self.model.getFrameId("F_striker_tip")
        J = pino.computeFrameJacobian(self.model, self.data, q, idx_, pino.LOCAL_WORLD_ALIGNED)[:3, :6]
        #null_J = scipy.linalg.null_space(J)
        pinvJ = np.linalg.pinv(J)
        alpha = (2. * np.random.random(6) - 1.) * Limits.q_dot
        null_proj = (np.eye(6) - pinvJ @ J)
        v_xyz_min = J @ (-Limits.q_dot - null_proj @ alpha)
        v_xyz_max = J @ (Limits.q_dot - null_proj @ alpha)
        v_xyz = (v_xyz_min + v_xyz_max) / 2. + (v_xyz_max - v_xyz_min) / 2. * (2.*np.random.random(3) - 1.) * v_xyz_scale
        q_dot = pinvJ @ v_xyz + null_proj @ alpha
        #q_dot = pinvJ @ v_xyz[:3] + null_J @ alpha
        v = J @ q_dot[:6]
        return q_dot, v

    def compute_q_ddot(self, q, q_dot, a_xyz_scale):
        q = np.pad(q, (0, self.n - q.shape[0]), mode='constant')
        q_dot = np.pad(q_dot, (0, self.n - q_dot.shape[0]), mode='constant')
        idx_ = self.model.getFrameId("F_striker_tip")
        pino.computeJointJacobiansTimeVariation(self.model, self.data, q, q_dot)
        dJ = pino.getFrameJacobianTimeVariation(self.model, self.data, idx_, pino.LOCAL_WORLD_ALIGNED)[:3, :6]
        J = pino.computeFrameJacobian(self.model, self.data, q, idx_, pino.LOCAL_WORLD_ALIGNED)[:3, :6]
        pinvJ = np.linalg.pinv(J)

        #th = np.pi * (2. * np.random.random() - 1.)
        #a_xyz = np.array([np.cos(th), np.sin(th), 0.01 * (2. * np.random.random() - 1.)])
        #alpha = 2. * np.random.random(3) - 1.
        #null_J = scipy.linalg.null_space(J)
        #q_ddot = pinvJ @ (a_xyz[:3] - dJ @ q_dot[:6]) + null_J @ alpha

        alpha = (2. * np.random.random(6) - 1.) * Limits.q_ddot
        null_proj = (np.eye(6) - pinvJ @ J)
        a_xyz_min = J @ (-Limits.q_ddot - null_proj @ alpha) + dJ @ q_dot[:6]
        a_xyz_max = J @ (Limits.q_ddot - null_proj @ alpha) + dJ @ q_dot[:6]
        a_xyz = (a_xyz_min + a_xyz_max) / 2. + (a_xyz_max - a_xyz_min) / 2. * (2.*np.random.random(3) - 1.) * a_xyz_scale
        q_ddot = pinvJ @ (a_xyz - dJ @ q_dot[:6]) + null_proj @ alpha
        return q_ddot
    
    def compute_q_ddot_for_a(self, q, q_dot, a):
        q = np.pad(q, (0, self.n - q.shape[0]), mode='constant')
        q_dot = np.pad(q_dot, (0, self.n - q_dot.shape[0]), mode='constant')
        idx_ = self.model.getFrameId("F_striker_tip")
        pino.computeJointJacobiansTimeVariation(self.model, self.data, q, q_dot)
        dJ = pino.getFrameJacobianTimeVariation(self.model, self.data, idx_, pino.LOCAL_WORLD_ALIGNED)[:3, :6]
        J = pino.computeFrameJacobian(self.model, self.data, q, idx_, pino.LOCAL_WORLD_ALIGNED)[:3, :6]
        pinvJ = np.linalg.pinv(J)
        alpha = (2. * np.random.random(6) - 1.) * Limits.q_ddot
        null_proj = (np.eye(6) - pinvJ @ J)
        q_ddot = pinvJ @ (a - dJ @ q_dot[:6]) + null_proj @ alpha
        return q_ddot

if __name__ == "__main__":
    urdf_path = "../../" + UrdfModels.striker
    q = np.array([0., 0.7135205165808098, 0., -0.5024774869152212, 0., 1.9256622406212651, 0., 0., 0.])
    v = np.array([1., 0.5, 0.])[:, np.newaxis]
    alpha = np.array([1., 0.5, 0.])[:, np.newaxis]
    vp = VelocityProjector(urdf_path)
    q_dot = vp.compute_q_dot(q, v, alpha)
    print(q_dot)
