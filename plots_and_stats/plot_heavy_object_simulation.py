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
    ("ours_l2048", "CNP-B (ours)"),
    ("cbirrt", "CBiRRT [11]"),
    ("nlopt", "TrajOpt [25]"),
    ("mpcmpnet", "MPC-MPNet [54]"),
    ("sst", "SST [41]"),
]
planners_names_map = {x: y for x, y in order}
planners_names_order = {x: i for i, (x, y) in enumerate(order)}
planners_names_map = dict(
                          ours_l2048="CNP-B (ours)",
                          cbirrt="CBiRRT [11]",
                          nlopt="TrajOpt [25]",
                          mpcmpnet="MPC-MPNet [54]",
                          sst="SST [41]",
                          )
planners_names_order = {x: i for i, x in enumerate(planners_names_map)}
colors_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray", "tab:olive",] #  "tab:cyan"]

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
            for dk, v in d.items():
                if k in v.keys() and v[k] is not None:
                    results[k][method].append(v[k])
            results[k][method] = np.array(results[k][method]).astype(np.float32)
    # positions = np.reshape(np.array([[i, i + spacing] for i in range(int(len(data) / 2.))]), -1)
    for k, v in results["alpha_beta"].items():
        results["alpha_beta"][k] = results["alpha_beta"][k][v != 0]
    labels = [planners_names_map[k] for k in planners_names_order]
    plt.rc('font', size=15)
    plt.rc('legend', fontsize=17)
    plt.figure(figsize=(18, 6))
    for i, (c, cv) in enumerate(categories.items()):
        ax = plt.subplot(1, 5, 1 + i)
        if cv.log:
            ax.set_yscale('log')
        ax.set_title(cv.title, rotation=0, ha="center", x=0.60)
        datapoints = [results[c][k] * cv.scale for k in planners_names_order]
        if c in ["valid", "finished"]:
            for j, p in enumerate(datapoints):
                lw = 2
                width = 0.7
                ax.bar(i + (width + lw * 0.005) * j, np.mean(p), width,
                       edgecolor=colors_list[j], fill=False, linewidth=lw,
                       hatch="//"
                       )
        else:
            bp = ax.boxplot(datapoints, positions=np.linspace(0., len(planners_names_order), len(planners_names_order)),
                            showmeans=True, showfliers=False,
                            meanprops=dict(marker="+", markeredgecolor="black"),
                            boxprops=dict(linewidth=2),
                            whiskerprops=dict(linewidth=2),
                            capprops=dict(linewidth=2),
                            widths=0.7)
            for i in range(len(bp["boxes"])):
                c = colors_list[i]
                bp["boxes"][i].set_color(c)
                bp["whiskers"][2 * i].set_color(c)
                bp["whiskers"][2 * i + 1].set_color(c)
                bp["caps"][2 * i].set_color(c)
                bp["caps"][2 * i + 1].set_color(c)
            ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks([])
    plt.subplots_adjust(hspace=0.3)
    plt.gcf().legend([x for x in bp["boxes"]], labels, ncol=len(labels), bbox_to_anchor=(0.8, 0.1),
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
for path in [os.path.join("../results/kino_exp/", k) for k in planners_names_map.keys()]:
    data = read_results(path)
    name = path.split("/")[-1]
    if name in planners_names_map.keys():
        results[name] = data

description = namedtuple("Description", "title scale log filter_finished")
categories = {
    "valid": description("Success ratio [%]", 100., False, False),
    "finished": description("Success ratio \n(no constraints) [%]", 100., False, False),
    "planning_time": description("Planning time [ms]", 1., True, False),
    "motion_time": description("Motion time [s]", 1., False, False),
    "alpha_beta": description("Vertical error [rad⋅s]", 1., True, False),
}
summary = {k: {} for k, v in categories.items()}
for name, data in results.items():
    for cat_k, cat_v in categories.items():
        summary[cat_k][name] = mean(data, cat_k, cat_v.filter_finished)

box_plot(results, categories)