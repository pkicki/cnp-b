import os
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(__file__)
PACKAGE_DIR = os.path.dirname(SCRIPT_DIR)
sys.path.append(PACKAGE_DIR)

from utils.bspline import BSpline
from utils.model import model_inference, load_model_kino

planner_path = os.path.join(PACKAGE_DIR, "trained_models/kino/last_n-150")
dim_q_control_points = 6
num_q_control_points = 15
num_t_control_points = 20
bsp = BSpline(num_q_control_points, num_T_pts=64)
bspt = BSpline(num_t_control_points, num_T_pts=64)
planner_model = load_model_kino(planner_path, num_q_control_points, bsp, bspt)

q_0 = np.zeros((7,))
q_d = np.ones((7,))

d = np.concatenate([q_0, q_d], axis=-1)[np.newaxis]
d = d.astype(np.float32)
q, dq, ddq, t, q_cps, t_cps = model_inference(planner_model, d, bsp, bspt)

plt.title("Robot configurations")
for i in range(6):
    ax = plt.subplot(321 + i)
    ax.set_title(rf"$q_{i}$")
    plt.plot(t, q[:, i])
plt.show()
