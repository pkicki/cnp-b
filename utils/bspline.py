import numpy as np
import os


class BSpline:
    def __init__(self, n, d=7, num_T_pts=1024, name=""):
        self.num_T_pts = num_T_pts
        self.d = d
        self.n_pts = n
        self.m = self.d + self.n_pts
        self.u = np.pad(np.linspace(0., 1., self.m + 1 - 2 * self.d), self.d, 'edge')
        fname = f"{os.path.dirname(__file__)}/bspline_{name}_{self.n_pts}_{self.d}_{self.num_T_pts}.npy"
        if os.path.exists(fname):
            d = np.load(fname, allow_pickle=True).item()
            self.N = d["N"]
            self.dN = d["dN"]
            self.ddN = d["ddN"]
            self.dddN = d["dddN"]
        else:
            self.N, self.dN, self.ddN, self.dddN = self.calculate_N()
            np.save(fname, {"N": self.N, "dN": self.dN, "ddN": self.ddN, "dddN": self.dddN}, allow_pickle=True)
        self.N = self.N.astype(np.float32)
        self.dN = self.dN.astype(np.float32)
        self.ddN = self.ddN.astype(np.float32)
        self.dddN = self.dddN.astype(np.float32)

    def calculate_N(self):
        def N(n, t, i):
            if n == 0:
                if self.u[i] <= t < self.u[i + 1]:
                    return 1
                else:
                    return 0
            s = 0.
            if self.u[i + n] - self.u[i] != 0:
                s += (t - self.u[i]) / (self.u[i + n] - self.u[i]) * N(n - 1, t, i)
            if self.u[i + n + 1] - self.u[i + 1] != 0:
                s += (self.u[i + n + 1] - t) / (self.u[i + n + 1] - self.u[i + 1]) * N(n - 1, t, i + 1)
            return s

        def dN(n, t, i):
            m1 = self.u[i + n] - self.u[i]
            m2 = self.u[i + n + 1] - self.u[i + 1]
            s = 0.
            if m1 != 0:
                s += N(n - 1, t, i) / m1
            if m2 != 0:
                s -= N(n - 1, t, i + 1) / m2
            return n * s

        def ddN(n, t, i):
            m1 = self.u[i + n] - self.u[i]
            m2 = self.u[i + n + 1] - self.u[i + 1]
            s = 0.
            if m1 != 0:
                s += dN(n - 1, t, i) / m1
            if m2 != 0:
                s -= dN(n - 1, t, i + 1) / m2
            return n * s

        def dddN(n, t, i):
            m1 = self.u[i + n] - self.u[i]
            m2 = self.u[i + n + 1] - self.u[i + 1]
            s = 0.
            if m1 != 0:
                s += ddN(n - 1, t, i) / m1
            if m2 != 0:
                s -= ddN(n - 1, t, i + 1) / m2
            return n * s

        T = np.linspace(0., 1., self.num_T_pts)
        Ns = [np.stack([N(self.d, t, i) for i in range(self.m - self.d)]) for t in T]
        Ns = np.stack(Ns, axis=0)
        Ns[-1, -1] = 1.
        dNs = [np.stack([dN(self.d, t, i) for i in range(self.m - self.d)]) for t in T]
        dNs = np.stack(dNs, axis=0)
        dNs[-1, -1] = (self.m - 2 * self.d) * self.d
        dNs[-1, -2] = -(self.m - 2 * self.d) * self.d
        ddNs = [np.stack([ddN(self.d, t, i) for i in range(self.m - self.d)]) for t in T]
        ddNs = np.stack(ddNs, axis=0)
        ddNs[-1, -1] = 2 * self.d * (self.m - 2 * self.d) ** 2 * (self.d - 1) / 2
        ddNs[-1, -2] = -3 * self.d * (self.m - 2 * self.d) ** 2 * (self.d - 1) / 2
        ddNs[-1, -3] = self.d * (self.m - 2 * self.d) ** 2 * (self.d - 1) / 2
        dddNs = [np.stack([dddN(self.d, t, i) for i in range(self.m - self.d)]) for t in T]
        dddNs = np.stack(dddNs, axis=0)
        dddNs[-1, -1] = 6 * self.d * (self.m - 2 * self.d) ** 3 * (self.d - 2)
        dddNs[-1, -2] = -10.5 * self.d * (self.m - 2 * self.d) ** 3 * (self.d - 2)
        dddNs[-1, -3] = 5.5 * self.d * (self.m - 2 * self.d) ** 3 * (self.d - 2)
        dddNs[-1, -4] = -self.d * (self.m - 2 * self.d) ** 3 * (self.d - 2)
        return Ns[np.newaxis], dNs[np.newaxis], ddNs[np.newaxis], dddNs[np.newaxis]