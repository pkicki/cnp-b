import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import namedtuple, OrderedDict

root_dir = os.path.dirname(__file__)
package_dir = os.path.dirname(root_dir)

planners_names_map = dict(ours_n9="9", ours_n12="12", ours_n15="15",
                          ours_n18="18", ours_n21="21")
colors = OrderedDict(ours_n9='tab:blue', ours_n12='tab:orange', ours_n15='tab:green', ours_n18='tab:red',
                     ours_n21='tab:purple')


def mean(r, k, abs=True):
    s = None
    if abs:
        s = [np.abs(v[k]) for _, v in r.items() if k in v and v[k] is not None]
    else:
        s = [v[k] for _, v in r.items() if k in v and v[k] is not None]
    return np.mean(s)


def box_plot(data, categories):
    results = {}
    for k in categories.keys():
        results[k] = {}
        for method, d in data.items():
            results[k][method] = []
            for v in d.values():
                if v[k] is not None:
                    results[k][method].append(v[k])
            results[k][method] = np.array(results[k][method]).astype(np.float32)
    labels = [planners_names_map[k] for k in colors.keys()]

    plt.rc('font', size=15)
    plt.rc('legend', fontsize=17)
    plt.figure(figsize=(11, 4))
    colors_list = list(colors.values())
    for i, (c, cv) in enumerate(categories.items()):
        ax = plt.subplot(1, 4, 1 + i)
        if cv.log:
            ax.set_yscale('log')
        ax.set_title(cv.title, rotation=0, ha="center", x=0.60, pad=10)
        datapoints = [results[c][k] * cv.scale for k in colors.keys()]
        if c in ["scored", "hit"]:
            for j, p in enumerate(datapoints):
                lw = 2
                width = 0.7
                ax.bar(i + (width + lw * 0.005) * j, np.mean(p), width,
                       edgecolor=colors_list[j], fill=False, linewidth=lw,
                       hatch="//"
                       )
            ax.set_ylim(95., 100.)
        else:
            bp = ax.boxplot(datapoints, positions=np.linspace(0., 0.3, len(colors)),
                            labels=labels, showmeans=True, showfliers=False,
                            meanprops=dict(marker="+", markeredgecolor="black"),
                            boxprops=dict(linewidth=2),
                            whiskerprops=dict(linewidth=2),
                            capprops=dict(linewidth=2),
                            widths=0.05)
            ax.set_xlim(-0.05, 0.35)
            for i in range(len(bp["boxes"])):
                c = list(colors.values())[i]
                bp["boxes"][i].set_color(c)
                bp["whiskers"][2 * i].set_color(c)
                bp["whiskers"][2 * i + 1].set_color(c)
                bp["caps"][2 * i].set_color(c)
                bp["caps"][2 * i + 1].set_color(c)
            ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])
    plt.subplots_adjust(wspace=0.3, top=0.8, bottom=0.15)
    plt.gcf().legend([x for x in bp["boxes"]], labels, ncol=len(labels), bbox_to_anchor=(0.8, 0.15),
                     frameon=False)
    plt.show()


def read_results(path):
    results = {}
    for p in glob(os.path.join(path, "*.res")):
        with open(p, 'rb') as fh:
            d = pickle.load(fh)
            name = p.split("/")[-1][:-4]
            xy = name.split("_")
            x = float(xy[0][1:])
            y = float(xy[1][1:])
            results[(x, y)] = d
    return results


results = {}
for path in [os.path.join("../results/hitting_exp/", k) for k in planners_names_map.keys()]:
    data = read_results(path)
    name = path.split("/")[-1]
    if name in planners_names_map.keys():
        results[name] = data

description = namedtuple("Description", "title scale log")
categories = {
    "scored": description("Score ratio [%]", 100., False),
    "puck_velocity_magnitude": description("Mean puck\n velocity [m/s]", 1., False),
    "actual_hitting_time": description("Mean hitting\n time [s]", 1., False),
    "actual_z_error": description("Z-axis error [mm⋅s]", 1000., False),
}
summary = {k: {} for k, v in categories.items()}
for name, data in results.items():
    for cat_k, cat_v in categories.items():
        summary[cat_k][name] = mean(data, cat_k)
a = 0

box_plot(results, categories)