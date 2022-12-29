import os
import sys
import numpy as np
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(__file__)
sys.path.append(SCRIPT_DIR)

from utils.bspline import BSpline
from utils.constants import Base
from utils.hpo_interface import get_hitting_configuration_opt
from utils.model import load_model_hitting, model_inference

planner_path = os.path.join(SCRIPT_DIR, "trained_models/hitting/best-149")
dim_q_control_points = 6
num_q_control_points = 15
num_t_control_points = 20
bsp = BSpline(num_q_control_points, num_T_pts=64)
bspt = BSpline(num_t_control_points, num_T_pts=64)
planner_model = load_model_hitting(planner_path, num_q_control_points, bsp, bspt)

q_0 = Base.configuration7
q_dot_0 = np.zeros_like(q_0)
q_ddot_0 = np.zeros_like(q_0)
x_hit = 1.0
y_hit = 0.2
th_hit = -0.1
q_d, q_dot_d = get_hitting_configuration_opt(x_hit, y_hit, Base.position[-1], th_hit, q_0)

d = np.concatenate([q_0, q_d, [x_hit, y_hit, th_hit], q_dot_0, q_ddot_0, q_dot_d], axis=-1)[np.newaxis]
d = d.astype(np.float32)
q, dq, ddq, t, q_cps, t_cps = model_inference(planner_model, d, bsp, bspt)

plt.title("Robot configurations")
for i in range(6):
    ax = plt.subplot(321 + i)
    ax.set_title(rf"$q_{i}$")
    plt.plot(t, q[:, i])
plt.show()
