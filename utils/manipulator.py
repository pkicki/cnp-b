from time import perf_counter

import numpy as np
import tensorflow as tf
import tensorflow_graphics.geometry.transformation as tfg
import pinocchio as pino
import xml.etree.ElementTree as ET


def make_S(x):
    if x.shape[-1] == 3:
        x1 = x[..., 0]
        x2 = x[..., 1]
        x3 = x[..., 2]
    elif x.shape[-2] == 3 and x.shape[-1] == 1:
        x1 = x[..., 0, 0]
        x2 = x[..., 1, 0]
        x3 = x[..., 2, 0]
    else:
        raise Exception(f"Wrong size of the tensor, obtained: {x.shape}, expected: [..., 3, 1] or [..., 3]")
    z = tf.zeros_like(x1)
    # populate rows of the skew-symmetric matrix
    r1 = tf.stack([z, -x3, x2], axis=-1)
    r2 = tf.stack([x3, z, -x1], axis=-1)
    r3 = tf.stack([-x2, x1, z], axis=-1)
    S = tf.stack([r1, r2, r3], axis=-2)
    return S


class LinkTF:
    def __init__(self, name, rpy, xyz, mass, inertia):
        self.name = name
        self.rpy = rpy
        self.xyz = np.array(xyz, dtype=np.float32)
        self.mass = mass
        self.inertia = inertia


class JointTF:
    def __init__(self, jtype, parent, child, rpy, xyz, axis, lb, ub):
        self.parent = parent
        self.child = child
        self.rpy = np.array(rpy, dtype=np.float32)
        self.roll = rpy[0]
        self.pitch = rpy[1]
        self.yaw = rpy[2]
        self.xyz = np.array(xyz, dtype=np.float32)
        self.axis = np.array(axis) if axis is not None else None
        # self.fixed = self.axis is None
        self.fixed = jtype == "fixed"
        self.lb = lb
        self.ub = ub
        self.Rb = tfg.rotation_matrix_3d.from_euler(self.rpy)

    def Rq(self, q):
        if self.fixed:
            R = tf.reshape(tf.eye(3), [1 for i in range(len(q.shape))] + [3, 3])
            return R
        else:
            return tfg.rotation_matrix_3d.from_euler(q[..., tf.newaxis] * self.axis)

    def R(self, q):
        if self.axis is None:
            return self.Rb
        Rq = self.Rq(q)
        return self.Rb @ Rq

    def T(self, q):
        R = self.R(q)
        Rp = tf.concat([R, np.array(self.xyz)[:, tf.newaxis]], axis=-1)
        T = tf.concat([Rp, np.array([0., 0., 0., 1.])[tf.newaxis]], axis=0)
        return T

    def Rp(self, q):
        return self.R(q), self.xyz[:, np.newaxis]


class Iiwa:
    def __init__(self, urdf_path):
        self.joints, self.links = Iiwa.parse_urdf(urdf_path)
        self.n_dof = len(self.joints)

    @staticmethod
    def parse_urdf(urdf_path):
        root = ET.parse(urdf_path).getroot()
        joints = []
        for joint in root.findall("joint"):
            jtype = joint.get('type')
            parent = joint.find("parent").get('link')
            child = joint.find("child").get('link')
            lb = float(joint.find("limit").get("lower")) if joint.find("limit") is not None else 0.0
            ub = float(joint.find("limit").get("upper")) if joint.find("limit") is not None else 0.0
            rpy = [float(x) for x in joint.find("origin").get('rpy').split()]
            xyz = [float(x) for x in joint.find("origin").get('xyz').split()]
            axis = joint.find("axis")
            if axis is not None:
                axis = [float(x) for x in axis.get('xyz').split()]
            joints.append(JointTF(jtype, parent, child, rpy, xyz, axis, lb, ub))
            # end at striker_tip
            if joints[-1].child.endswith("striker_tip"):
                break
        links = []
        for link in root.findall("link"):
            name = link.get("name")
            inertial = link.find("inertial")
            if inertial is None:
                continue
            rpy = [float(x) for x in inertial.find("origin").get('rpy').split()]
            xyz = [float(x) for x in inertial.find("origin").get('xyz').split()]
            mass = float(inertial.find("mass").get("value"))
            I = {k: float(v) for k, v in inertial.find("inertia").items()}
            inertia = np.array(
                [[I["ixx"], I["ixy"], I["ixz"]], [I["ixy"], I["iyy"], I["iyz"]], [I["ixz"], I["iyz"], I["izz"]]])
            links.append(LinkTF(name, rpy, xyz, mass, inertia))
        return joints, links

    def forward_kinematics_R_list(self, q):
        q = tf.concat([q, tf.zeros_like(q)[..., :7 - q.shape[-1]]], axis=-1)
        qs = []
        qidx = 0
        for i in range(self.n_dof):
            if self.joints[i].fixed:
                qs.append(tf.zeros_like(q)[..., 0])
            else:
                qs.append(q[..., qidx])
                qidx += 1

        q = tf.stack(qs, axis=-1)
        q = tf.cast(q, tf.float32)
        Racc = tf.eye(3, batch_shape=tf.shape(q)[:-1])
        z = tf.zeros_like(q[..., 0])
        xyz = tf.stack([z, z, z], axis=-1)[..., tf.newaxis]
        xyz_list = [xyz]
        R_list = [Racc]
        for i in range(self.n_dof):
            qi = q[..., i]
            j = self.joints[i]
            R, p = j.Rp(qi)
            for i in range(len(tf.shape(q)) - 1):
                p = p[tf.newaxis]
            xyz = xyz + Racc @ p
            Racc = Racc @ R
            xyz_list.append(xyz)
            R_list.append(Racc)
        return xyz_list, R_list

    def forward_kinematics(self, q):
        return self.forward_kinematics_R_list(q)[0][-1]

    def forward_kinematics_R(self, q):
        xyzs, Rs = self.forward_kinematics_R_list(q)
        return xyzs[-1], Rs[-1]

    def interpolate_links(self, xyzs):
        # dists = np.linalg.norm(np.diff(np.concatenate(xyzs, axis=-1), axis=-1), axis=-2)
        xyzs_ = [xyzs[0]]
        # iiwa_cup
        #for i, n in enumerate([0, 1, 2, 2, 2, 1, 2, 0, 0, 1]):
        # iiwa
        for i, n in enumerate([0, 1, 2, 2, 2, 1, 2, 0, 0]):
            s = tf.linspace(0., 1., n + 2)[1:]
            for x in s:
                xyzs_.append(x * xyzs[i + 1] + (1. - x) * xyzs[i])
        xyzs_interp = tf.stack(xyzs_, axis=-3)
        # dists_ = np.linalg.norm(np.diff(np.concatenate(xyzs_interp, axis=-1), axis=-1), axis=-2)
        return xyzs_interp

    def interpolated_forward_kinematics(self, q):
        xyzs, Rs = self.forward_kinematics_R_list(q)
        return self.interpolate_links(xyzs), Rs[-1]

    def rnea(self, q, dq, ddq):
        # todo should be rewritten in order to work also if fixed joints are present in the middle of the kinematic chain
        q = tf.concat([q, tf.zeros_like(q)[..., :7 - q.shape[-1]]], axis=-1)
        dq = tf.concat([dq, tf.zeros_like(dq)[..., :7 - dq.shape[-1]]], axis=-1)
        ddq = tf.concat([ddq, tf.zeros_like(ddq)[..., :7 - ddq.shape[-1]]], axis=-1)
        #qs = []
        #dqs = []
        #ddqs = []
        #qidx = 0
        #for i in range(self.n_dof):
        #    if self.joints[i].fixed:
        #        qs.append(tf.zeros_like(q)[..., 0])
        #        dqs.append(tf.zeros_like(q)[..., 0])
        #        ddqs.append(tf.zeros_like(q)[..., 0])
        #    else:
        #        qs.append(q[..., qidx])
        #        dqs.append(dq[..., qidx])
        #        ddqs.append(ddq[..., qidx])
        #        qidx += 1
        #q = tf.cast(tf.stack(qs, axis=-1), tf.float32)
        #dq = tf.cast(tf.stack(dqs, axis=-1), tf.float32)
        #ddq = tf.cast(tf.stack(ddqs, axis=-1), tf.float32)

        one = np.ones_like(q[..., 0])
        zero = np.zeros_like(q[..., 0])
        omega_0_0 = tf.stack([zero, zero, zero], axis=-1)[..., tf.newaxis]
        omega_i_i = [omega_0_0]
        alpha_0_0 = tf.stack([zero, zero, zero], axis=-1)[..., tf.newaxis]
        alpha_i_i = [alpha_0_0]
        ae_0_0 = tf.stack([zero, zero, zero], axis=-1)[..., tf.newaxis]
        ae_i_i = [ae_0_0]
        ac_0_0 = tf.stack([zero, zero, zero], axis=-1)[..., tf.newaxis]
        ac_i_i = [ac_0_0]
        Racc = [tf.eye(3, batch_shape=tf.shape(q)[:-1])]
        r_i_ci = []
        r_ip1_ci = []
        for i in range(1, self.n_dof - 1):
            j = self.joints[i]
            jp1 = self.joints[i + 1]
            l = self.links[i]
            qi = q[..., i - 1]
            dqi = dq[..., i - 1]
            ddqi = ddq[..., i - 1]
            dq_i_i = tf.stack([zero, zero, dqi], axis=-1)[..., tf.newaxis]
            ddq_i_i = tf.stack([zero, zero, ddqi], axis=-1)[..., tf.newaxis]
            R = j.R(qi)
            Racc.append(Racc[-1] @ R)
            RT = tf.linalg.matrix_transpose(R)
            omega_i_i_ = RT @ omega_i_i[-1] + dq_i_i
            omega_i_i.append(omega_i_i_)
            S_omega_i_i = make_S(omega_i_i_)
            alpha_i_i_ = RT @ alpha_i_i[-1] + ddq_i_i + S_omega_i_i @ dq_i_i
            alpha_i_i.append(alpha_i_i_)
            S_alpha_i_i = make_S(alpha_i_i_)
            r_i_ci_ = (l.xyz * one[..., tf.newaxis])[..., tf.newaxis]
            ac_i_i_ = RT @ ae_i_i[-1] + S_alpha_i_i @ r_i_ci_ + S_omega_i_i @ S_omega_i_i @ r_i_ci_
            ac_i_i.append(ac_i_i_)
            r_i_ip1 = jp1.xyz[tf.newaxis, tf.newaxis, :, tf.newaxis]
            r_ip1_ci_ = r_i_ci_ - r_i_ip1
            ae_i_i_ = RT @ ae_i_i[-1] + S_alpha_i_i @ r_i_ip1 + S_omega_i_i @ S_omega_i_i @ r_i_ip1
            ae_i_i.append(ae_i_i_)
            r_i_ci.append(r_i_ci_)
            r_ip1_ci.append(r_ip1_ci_)

        f_0 = tf.stack([zero, zero, zero], axis=-1)[..., tf.newaxis]
        f_i = [f_0]
        t_0 = tf.stack([zero, zero, zero], axis=-1)[..., tf.newaxis]
        t_i = [t_0]
        g_0 = tf.stack([zero, zero, -9.81 * one], axis=-1)[..., tf.newaxis]
        t_i_joints = []
        for i in range(1, self.n_dof - 1):
            jp1 = self.joints[-i]  # check indexes if valid
            j = self.joints[-i - 1]  # check indexes if valid
            l = self.links[-i - 1]  # check indexes if valid
            R_ip1_i = jp1.Rb
            if i > 2:
                qip2 = q[..., -i + 1]
                Rq = self.joints[-i + 1].Rq(qip2)
                R_ip1_i = R_ip1_i @ Rq
            R_0_i = tf.linalg.matrix_transpose(Racc[-i])  # not sure if -i or -i-1
            g_i = R_0_i @ g_0
            m_i = l.mass
            I_i = l.inertia[tf.newaxis, tf.newaxis]
            f_ip1_i = R_ip1_i @ f_i[-1]
            f_i_ = m_i * ac_i_i[-i] + f_ip1_i - m_i * g_i
            f_i.append(f_i_)
            t_i_ = R_ip1_i @ t_i[-1] - make_S(f_i_) @ r_i_ci[-i] + make_S(f_ip1_i) @ r_ip1_ci[-i] + I_i @ alpha_i_i[-i] \
                   + make_S(omega_i_i[-i]) @ I_i @ omega_i_i[-i]
            t_i.append(t_i_)
            if j.axis is not None:
                t_i_joint = (one[..., tf.newaxis, tf.newaxis] * np.array(j.axis) @ t_i_)[..., 0, 0]
                t_i_joints.append(t_i_joint)

        f_i = f_i[::-1]
        t_i = t_i[::-1]
        t_i_joints = t_i_joints[::-1]
        t_i_joints = tf.stack(t_i_joints, axis=-1)

        return t_i_joints


if __name__ == "__main__":
    urdf_path = "../iiwa.urdf"
    pino_model = pino.buildModelFromUrdf(urdf_path)
    pino_data = pino_model.createData()
    man = Iiwa(urdf_path)

    # FK test
    # qk = np.zeros(6)
    # qk = np.array([0.2, 0.2, 0.5, 0.9, 0., 0.])
    # q = np.concatenate([qk, np.zeros(3)], axis=-1)
    # pino.forwardKinematics(pino_model, pino_data, q)
    # xyz_pino = pino_data.oMi[-1].translation

    # xyz = man.forward_kinematics(qk)
    # print(xyz_pino)
    # print(xyz.numpy()[..., 0])

    q = np.array([0., 1., 0., -1., 1., 0.], dtype=np.float32)
    # q = np.array([0., 0., 0., 0., 0., 0.], dtype=np.float32)
    dq = np.array([1., 1., 1., 1., 1., 1.], dtype=np.float32)
    # dq = np.array([0., 0., 0., 0., 0., 0.], dtype=np.float32)
    ddq = np.array([1., 0., -1., 0., 1., 1.], dtype=np.float32)

    q_ = q[np.newaxis, np.newaxis]
    dq_ = dq[np.newaxis, np.newaxis]
    ddq_ = ddq[np.newaxis, np.newaxis]

    q_ = tf.tile(q_, (128, 1024, 1))
    dq_ = tf.tile(dq_, (128, 1024, 1))
    ddq_ = tf.tile(ddq_, (128, 1024, 1))
    tau = man.rnea(q_, dq_, ddq_)
    t0 = perf_counter()
    tau = man.rnea(q_, dq_, ddq_)
    t1 = perf_counter()
    a = 0
    print("TF:")
    print(tau.numpy()[0, 0])
    print("TIME:", t1 - t0)

    q = np.concatenate([q, np.zeros(1)], axis=-1)
    dq = np.concatenate([dq, np.zeros(1)], axis=-1)
    ddq = np.concatenate([ddq, np.zeros(1)], axis=-1)
    t0 = perf_counter()
    for i in range(128 * 1024):
        ret = pino.rnea(pino_model, pino_data, q, dq, ddq)
    t1 = perf_counter()
    b = 0
    print("PINO:")
    print(ret)
    print("TIME:", t1 - t0)
    # xyz_pino = pino_data.oMi[-1]
