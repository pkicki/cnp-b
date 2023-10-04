import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import namedtuple

root_dir = os.path.dirname(__file__)
package_dir = os.path.dirname(root_dir)

masses = [6, 8, 10, 12, 14, 16, 18]
planners_names = [f"ours_l0064_{i}kg" for i in masses]
planners_names_map = {pn: float(pn.split("_")[2][:-2]) for pn in planners_names}



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

    plt.rc('font', size=17)
    plt.rc('legend', fontsize=17)
    plt.rc('xtick', labelsize=17)
    plt.figure(figsize=(12, 5))
    for i, (c, cv) in enumerate(categories.items()):
        ax = plt.subplot(1, 3, 1 + i)
        if cv.log:
            ax.set_yscale('log')
        ax.set_title(cv.title, rotation=0, ha="center", x=0.60)
        ax.set_xlabel("Object mass [kg]")
        datapoints = [results[c][k] * cv.scale for k in planners_names]
        color = "b"
        if c in ["valid", "finished"]:
            for j, p in enumerate(datapoints):
                lw = 2
                width = 1.0
                ax.bar(masses[j], np.mean(p), width,
                       edgecolor=color, fill=False, linewidth=lw,
                       hatch="//"
                       )
        else:
            bp = ax.boxplot(datapoints, positions=np.array(masses),
                            showmeans=True, showfliers=False,
                            meanprops=dict(marker="+", markeredgecolor="black"),
                            boxprops=dict(linewidth=2, color=color),
                            whiskerprops=dict(linewidth=2, color=color),
                            capprops=dict(linewidth=2, color=color),
                            widths=1.0)
            ax.set_xlim(5, 19)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_xticks(masses)
        ax.set_xticklabels(masses)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                     ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(20)
    plt.subplots_adjust(hspace=0.5, bottom=0.2, wspace=0.25)
    plt.show()


def read_results(path):
    results = {}
    for p in glob(os.path.join(path, "*.res")):
        with open(p, 'rb') as fh:
            d = pickle.load(fh)
            name = p.split("/")[-1][:-4]
            results[name] = d
    return results


results = {}
for path in [os.path.join("../results/kino_exp/", k) for k in planners_names_map.keys()]:
    data = read_results(path)
    name = path.split("/")[-1]
    if name in planners_names_map.keys():
        results[name] = data

description = namedtuple("Description", "title scale log")
categories = {
    "valid": description("Success ratio [%]", 100., False),
    "motion_time": description("Mean motion time [s]", 1., False),
    "alpha_beta": description("Mean vertical error [rad]", 1., False),
}
summary = {k: {} for k, v in categories.items()}
for name, data in results.items():
    for cat_k, cat_v in categories.items():
        summary[cat_k][name] = mean(data, cat_k)

box_plot(results, categories)