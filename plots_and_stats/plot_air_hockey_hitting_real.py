import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import namedtuple

root_dir = os.path.dirname(__file__)
print(root_dir)
package_dir = os.path.dirname(root_dir)
baseline_path = "../results/hitting_exp/baseline_opt_lp/"
ours_path = "../results/hitting_exp/ours_opt_lp/"
print(ours_path)

def box_plot(data):
    extract = lambda x, y: [v[y] for _, v in x.items() if y in v]
    description = namedtuple("Description", "name title scales")
    descriptions = [
                    description("actual_hitting_time", "Hitting \n time [s]", (1., 1.)),
                    description("puck_velocity_magnitude", "Puck\n velocity [m/s]", (1., 1.)),
                    description("planned_z_error", "Planned z-axis \n error [mm⋅s]", (1000., 1000.)),
                    description("actual_z_error", "Z-axis\n error [mm⋅s]", (1000., 1000.)),
                    description("joint_trajectory_error", "Joint trajectory \n error [rad⋅s]", (1., 1.)),
                    ]
    collection = [[np.array(extract(d, k)) for d in data] for k in [x.name for x in descriptions]]
    collection[0][1] = np.abs(collection[0][1])

    spacing = 0.2
    c1 = "red"
    c2 = "blue"

    plt.rc('font', size=15)
    plt.rc('legend', fontsize=17)
    plt.figure(figsize=(9, 7))
    for i in range(len(descriptions)):
        ax = plt.subplot(1, 5, 1 + i)
        ax.set_title(descriptions[i].title, rotation=0, ha="center", x=0.60)
        datapoints = [x * descriptions[i].scales[k] for k, x in enumerate(collection[i])]
        bp = ax.boxplot(datapoints, positions=np.arange(0., spacing*len(collection[i]), spacing), labels=["CNP-B (ours)", "AQP [8]"],
                        widths=0.15)
        ax.set_xlim(-0.15, 0.35)
        for i in range(len(bp["boxes"])):
            if i % 2:
                bp["boxes"][i].set_color(c1)
                bp["fliers"][i].set_color(c1)
                bp["whiskers"][2 * i].set_color(c1)
                bp["whiskers"][2 * i + 1].set_color(c1)
                bp["caps"][2 * i].set_color(c1)
                bp["caps"][2 * i + 1].set_color(c1)
            else:
                bp["boxes"][i].set_color(c2)
                bp["fliers"][i].set_color(c2)
                bp["whiskers"][2 * i].set_color(c2)
                bp["whiskers"][2 * i + 1].set_color(c2)
                bp["caps"][2 * i].set_color(c2)
                bp["caps"][2 * i + 1].set_color(c2)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
    plt.subplots_adjust(hspace=0.3)
    plt.gcf().legend([bp["boxes"][0], bp["boxes"][1]], ['CNP-B (ours)', 'AQP [8]'], ncol=2, bbox_to_anchor=(0.7, 0.1), frameon=False)
    plt.show()


def scoring_ratio(r):
    return mean(r, "scored")


def mean(r, k, abs=True):
    s = [v[k] for _, v in r.items() if k in v] if not abs else [np.abs(v[k]) for _, v in r.items() if k in v]
    return np.mean(s)


def plot_scatter(r1, r2, k, abs=True, name="", alpha=1.):
    xy = np.array([v["puck_initial_pose"][:2] for _, v in r1.items()])
    v1 = [np.abs(v[k]) if k in v else 0. for _, v in r1.items()] if abs else \
        [v[k] if k in v else 0. for _, v in r1.items()]
    v2 = [np.abs(v[k]) if k in v else 0. for _, v in r2.items()] if abs else \
        [v[k] if k in v else 0. for _, v in r2.items()]
    v1 = np.array(v1)
    v2 = np.array(v2)
    def color(v):
        return np.array([[1., 0., 0.] if x == 0 else [0., 1., 0.] for x in v])
    def marker(v):
        return np.array(["s" if x == 0 else "o" for x in v])
    ax = plt.subplot(121)
    ax.set_title("CNP-B")
    plt.scatter(xy[v1 == 0, 0], xy[v1 == 0, 1], c="r", marker="o")
    plt.scatter(xy[v1 == 1, 0], xy[v1 == 1, 1], c="g", marker="s")
    plt.xlim(0.6, 1.5)
    plt.ylim(-0.45, 0.45)
    ax = plt.subplot(122)
    ax.set_title("AQP")
    plt.scatter(xy[v2 == 0, 0], xy[v2 == 0, 1], c="r", marker="o")
    plt.scatter(xy[v2 == 1, 0], xy[v2 == 1, 1], c="g", marker="s")
    #plt.scatter(xy[:, 0], xy[:, 1], c=color(v2), marker=marker(v2))
    #plt.colorbar()
    #plt.clim(0.5, 2.2)
    plt.xlim(0.6, 1.5)
    plt.ylim(-0.45, 0.45)
    plt.show()


def read_results(path):
    results = {}
    for p in glob(path + "*.res"):
        with open(p, 'rb') as fh:
            d = pickle.load(fh)
            results[p[-6:-4]] = d
    return results


ours = read_results(ours_path)
baseline = read_results(baseline_path)
print("SCORING RATIOS:")
print("OURS:", scoring_ratio(ours))
print("BASELINE:", scoring_ratio(baseline))
box_plot((ours, baseline))
plot_scatter(ours, baseline, "scored")