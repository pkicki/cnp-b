import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import namedtuple
import matplotlib

matplotlib.rcParams['mathtext.fontset'] = 'cm'

root_dir = os.path.dirname(__file__)
package_dir = os.path.dirname(root_dir)

order = [
    ("ours", "CNP-B (ours)"),
    ("bitstar_0s", "BIT*"),
    ("bitstar_1s_noprune", "BIT* 1s"),
    ("bitstar_10s_noprune", "BIT* 10s"),
]
planners_names_map = {x: y for x, y in order}
planners_names_order = {x: i for i, (x, y) in enumerate(order)}
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
colors = colors[:len(order)]


def mean(r, k, filter_finished):
    if filter_finished:
        s = [v[k] for _, v in r.items() if (k in v and v[k] is not None and v["finished"])]
    else:
        s = [v[k] for _, v in r.items() if k in v and v[k] is not None]
    return np.mean(s)


def box_plot(data, categories):
    results = {}
    for k in categories.keys():
        results[k] = {}
        for method, d in data.items():
            results[k][method] = []
            for dk, dv in d.items():
                if k in dv.keys():
                    results[k][method].append(dv[k])
            results[k][method] = np.array(results[k][method]).astype(np.float32)
    plt.rc('font', size=15)
    plt.rc('legend', fontsize=17)
    plt.figure(figsize=(9, 7))
    for i, (c, cv) in enumerate(categories.items()):
        ax = plt.subplot(1, 3, 1 + i)
        if cv.log:
            ax.set_yscale('log')
        ax.set_title(cv.title, rotation=0, ha="center", x=0.60)
        datapoints = [results[c][k] * cv.scale for k in planners_names_map.keys()]
        if c in ["valid", "finished"]:
            for j, p in enumerate(datapoints):
                lw = 2
                width = 0.1
                ax.bar(i + (width + lw * 0.005) * j, np.mean(p), width,
                       edgecolor=colors[j], fill=False, linewidth=lw,
                       hatch="//"
                       )
                plt.ylim(99, 100)
        else:
            bp = ax.boxplot(datapoints, positions=np.linspace(0., 0.4, len(order)),
                            showmeans=True, showfliers=False,
                            meanprops=dict(marker="+", markeredgecolor="black"),
                            boxprops=dict(linewidth=2),
                            whiskerprops=dict(linewidth=2),
                            capprops=dict(linewidth=2),
                            widths=0.05)
            ax.set_xlim(-0.05, 0.45)
            for i in range(len(bp["boxes"])):
                c = colors[i]
                bp["boxes"][i].set_color(c)
                bp["whiskers"][2 * i].set_color(c)
                bp["whiskers"][2 * i + 1].set_color(c)
                bp["caps"][2 * i].set_color(c)
                bp["caps"][2 * i + 1].set_color(c)
            ax.spines['bottom'].set_visible(False)
        ax.set_xticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.subplots_adjust(hspace=0.3, bottom=0.2)
    plt.gcf().legend([x for x in bp["boxes"]], planners_names_map.values(), ncol=len(colors),
                     bbox_to_anchor=(0.9, 0.15),
                     frameon=False)
    plt.show()


def read_results(path):
    results = {}
    for p in glob(os.path.join(path, "*.res")):
        with open(p, 'rb') as fh:
            d = pickle.load(fh)
            name = p.split("/")[-1][:-4]
            idx = int(name)
            results[idx] = d
    return results


results = {}
for path in [os.path.join("../results/obs_exp/", k) for k in planners_names_map.keys()]:
    # for path in glob("/home/piotr/b8/ah_ws/results/obs_exp/*"):
    data = read_results(path)
    name = path.split("/")[-1]
    if name in planners_names_map.keys():
        results[name] = data

description = namedtuple("Description", "title scale log filter_finished")
categories = {
    "valid": description("Success ratio [%]", 100., False, False),
    "planning_time": description("Planning time [ms]", 1., True, False),
    "motion_time": description("Motion time [s]", 1., False, False),
}
summary = {k: {} for k, v in categories.items()}
for name, data in results.items():
    for cat_k, cat_v in categories.items():
        summary[cat_k][name] = mean(data, cat_k, cat_v.filter_finished)

box_plot(results, categories)
