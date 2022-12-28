import matplotlib.pyplot as plt
from utils.constants import TableConstraint


class Table:
    def __init__(self):
        self.xlb = TableConstraint.XLB
        self.ylb = TableConstraint.YLB
        self.xrt = TableConstraint.XRT
        self.yrt = TableConstraint.YRT
        self.z = TableConstraint.Z

    def plot(self):
        plt.plot([self.xlb, self.xlb, self.xrt, self.xrt, self.xlb],
                 [self.ylb, self.yrt, self.yrt, self.ylb, self.ylb], 'b')

    def inside(self, x, y):
        return self.xlb <= x <= self.xrt and self.ylb <= y <= self.yrt