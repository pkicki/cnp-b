import os
import sys
from time import perf_counter

import numpy as np
import pinocchio as pino

SCRIPT_DIR = os.path.dirname(__file__)
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PACKAGE_DIR)

from utils.constants import Limits, UrdfModels, TableConstraint, Base
import matplotlib.pyplot as plt

import utils.hpo_opt_new as hpoo


def get_hitting_configuration_opt(x, y, z, th, q0=None):
    if q0 is None:
        q0 = Base.configuration + [0.]
    q0 = q0 + [0.] * (9 - len(q0))
    s = hpoo.optimize(x, y, z, np.cos(th), np.sin(th), q0)
    if not s:
        return None, None
    q = s[:7]
    q_dot = np.array(s[9:16])
    return q, q_dot.tolist()

if __name__ == "__main__":
    q0_ = [0., 0.7135, 0., -0.5025, 0., 1.9257, 0.]
    #q0 = [0.0, 0.06580,  0.0, -1.45996, 0.,  1.22487, 0.0]
    q0 = [0.0, 0.06811,  0.0, -1.48, 0.,  1.2544, 0.0]
    q0 = Base.configuration + [0.]
    q0_ = Base.configuration + [0.]
    urdf_path = os.path.join(os.path.dirname(__file__), "..", UrdfModels.striker)
    model = pino.buildModelFromUrdf(urdf_path)
    data = model.createData()
    x = 0.95
    y = -0.41
    th = 0.24
    print(np.cos(th), np.sin(th))
    q, q_dot = get_hitting_configuration_opt(x, y, TableConstraint.Z, th, q0)
    q = np.concatenate([np.array(q), np.zeros(2)], axis=-1)
    pino.forwardKinematics(model, data, q)
    xyz_pino = data.oMi[-1].translation
    idx_ = model.getFrameId("F_striker_tip")
    J = pino.computeFrameJacobian(model, data, q, idx_, pino.LOCAL_WORLD_ALIGNED)[:3, :6]
    print()
    print("J:", J.shape)
    print()
    print(xyz_pino)
    print()
    print("Q:", q)
    #print(q_dot)
    #q0 = [0., 0.7135, 0., -0.5025, 0., 1.9257, 0.]
    print(np.sum(np.abs(np.array(q0)[:6] - np.array(q)[:6]) / Limits.q_dot))
    #assert False
    v = J @ np.array(q_dot)[:6]

    X, Y = 15, 31
    x = 1.
    th = 0.1
    xs = np.linspace(0.7, 1.3, X)
    #ths = np.linspace(-0.2, 0.2, X)
    #xs = [1.1]
    ths = [0.0]
    ys = np.linspace(-0.4, 0.4, Y)
    qi0s = []
    times = []
    #q0 = [-0.1199166764301418, 0.925443585720224, -0.09106010491374625, -0.3463669629041281, -0.08702749577145877, 1.3843531121676345, 0.0]
    vdiffs = []
    vmaxdiffs = []
    q0diffs = []
    #qdiffs = np.zeros((X, Y, 6))
    #qs = np.zeros((X, Y, 7))
    qdiffs = []
    qs = []
    q0s = []
    vs = []
    ccs = []
    Ymid = int(len(ys)/2)
    #for i in range(len(xs)):
    for x in xs:
        #for j in range(Ymid):
        for y in ys:
            for th in ths:
                #q0 = q0_
                for sign in [1]:
                #for sign in [-1, 1]:
                    #x = xs[i]
                    #jidx = Ymid + sign * j
                    #y = ys[jidx]
                    #print("X", x)
                    #print("Y", y)
                    #print("Q0", q0)
                    t0 = perf_counter()
                    #q0[0] = y / 2

                    #q0[0] = y * 3 / 4
                    #q0[3] = 0.5 * (x - 0.9) + 0.1
                    #q0[2] = y / 4
                    #q0[4] = y / 4
                    #q0[6] = q0[6] + y / 4

                    q0s.append(q0.copy())
                    #q, q_dot, mag, v = get_hitting_configuration(x, y, TableConstraint.Z, th, q0)
                    q, q_dot = get_hitting_configuration_opt(x, y, TableConstraint.Z, th, q0)
                    if q is None:
                        q = np.zeros((7)).tolist()
                        q_dot = np.zeros((7)).tolist()
                    #q_, q_dot_, mag_opt, v_opt = get_hitting_configuration_opt(x, y, TableConstraint.Z, th, q0)
                    idx_ = model.getFrameId("F_striker_tip")
                    q_ = np.array(q + [0., 0.])
                    J = pino.computeFrameJacobian(model, data, q_, idx_, pino.LOCAL_WORLD_ALIGNED)[:3, :6]
                    v = J @ q_dot[:6]
                    dq = np.concatenate([np.array(q_dot), np.zeros(2)], axis=-1)
                    ddq = np.zeros_like(dq)
                    gcc = pino.rnea(model, data, q_, dq, ddq)
                    g = pino.rnea(model, data, q_, ddq, ddq)
                    cc = gcc - g
                    ccs.append(np.sum(np.abs(cc)))

                    vs.append(v)
                    qs.append(q[:6])
                    #q0 = q
                    t1 = perf_counter()
                    times.append(t1 - t0)
                    qi0s.append(q[0])
                    vdiffs.append(np.sum(np.abs(np.array(q0) - np.array(q))[:6] / Limits.q_dot))
                    vmaxdiffs.append(np.max(np.abs(np.array(q0) - np.array(q))[:6] / Limits.q_dot))
                    q0diffs.append(np.abs(np.array(q0) - np.array(q))[0])
                    qdiffs.append(np.abs(np.array(q0_) - np.array(q))[:6])
                    #qdiffs[i, jidx] = np.abs(np.array(q0) - np.array(q))[:6]
                    #qs[i, jidx] = q
                    #q0[:7] = q[:7]
    print()
    print("MEAN T: ", np.mean(times))
    print(np.std(times))
    print(np.min(times))
    print("MAX T: ", np.max(times))
    print("MAX Q0DIFF: ", np.max(q0diffs))
    q0s = np.array(q0s)
    qs = np.array(qs)

    vs = np.array(vs)
    vmags = np.linalg.norm(vs, axis=-1)
    ccs = np.array(ccs)
    vdiffs = np.array(vdiffs)
    q0diffs = np.array(q0diffs)
    qdiffs = np.array(qdiffs)
    vmags = np.reshape(np.reshape(vmags, (X, Y)).T, -1)
    ccs = np.reshape(np.reshape(ccs, (X, Y)).T, -1)
    vdiffs = np.reshape(np.reshape(vdiffs, (X, Y)).T, -1)
    q0diffs = np.reshape(np.reshape(q0diffs, (X, Y)).T, -1)
    qdiffs = np.reshape(np.transpose(np.reshape(qdiffs, (X, Y, 6)), (1, 0, 2)), (-1, 6))
    qs = np.reshape(np.transpose(np.reshape(qs, (X, Y, 6)), (1, 0, 2)), (-1, 6))
    #qdiffs = np.reshape(np.transpose(np.reshape(qdiffs, (X, Y, 6)), (0, 1, 2)), (-1, 6))
    x, y = np.meshgrid(xs, ys)
    #y, th = np.meshgrid(ys, ths)
    #x = np.reshape(x, -1)
    #x = np.reshape(np.reshape(x, (X, Y)).T, -1)
    #y = np.reshape(np.reshape(y, (X, Y)).T, -1)
    x = np.reshape(np.reshape(x, (X, Y)), -1)
    y = np.reshape(np.reshape(y, (X, Y)), -1)
    #th = np.reshape(np.reshape(th, (X, Y)).T, -1)
    #plt.scatter(x, y, c=vdiffs)
    plt.scatter(x, y, c=vmags, vmin=0.7, vmax=2.1)
    #plt.scatter(y, th, c=vdiffs)
    plt.colorbar()
    plt.show()
    plt.scatter(x, y, c=ccs)
    #plt.scatter(y, th, c=q0diffs)
    plt.colorbar()
    plt.show()
    for i in range(6):
        plt.subplot(231 + i)
        #plt.scatter(y, th, c=qdiffs[..., i])
        plt.scatter(x, y, c=qdiffs[..., i])
        plt.colorbar()
    plt.show()
    for i in range(6):
        plt.subplot(231 + i)
        #plt.scatter(y, th, c=qdiffs[..., i])
        plt.scatter(x, y, c=qs[..., i])
        plt.colorbar()
    plt.show()


    #plt.subplot(331)
    #plt.plot(ys, qi0s)
    #plt.plot(ys, vdiffs)
    #plt.plot(ys, vmaxdiffs)
    #for i in range(6):
    #    plt.subplot(332 + i)
    #    plt.plot(ys, qs[:, i])
    #plt.show()

