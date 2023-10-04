import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import namedtuple

root_dir = os.path.dirname(__file__)
package_dir = os.path.dirname(root_dir)

heights = np.arange(-20, 40+1, 2).tolist()
planners_ref_names = [f"baseours_h{i}_7jc" for i in heights]
planners_par_names = [f"newours_h{i}_7jc" for i in heights]
planners_ref_names_map = {pn: float(pn.split("_")[1][1:]) for pn in planners_ref_names}
planners_par_names_map = {pn: float(pn.split("_")[1][1:]) for pn in planners_par_names}



def mean(r, k, abs=True):
    s = None
    if abs:
        s = [np.abs(v[k]) for _, v in r.items() if k in v and v[k] is not None]
    else:
        s = [v[k] for _, v in r.items() if k in v and v[k] is not None]
    return np.mean(s)


def plot_datapoints(ax, c, datapoints, color, style):
    if c in ["scored", "hit"]:
        ax.plot(heights, np.mean(datapoints, axis=-1), color + style)
        #for j, p in enumerate(datapoints):
        #    lw = 2
        #    width = 1.0
        #    ax.bar(heights[j], np.mean(p), width,
        #           edgecolor=color, fill=False, linewidth=lw,
        #           hatch="//"
        #           )
        #    # ax.set_xlim(-0.05, 0.45)
    else:
        th = 3
        bp = ax.boxplot(datapoints, positions=np.array(heights),
                        # labels=labels, showmeans=True, showfliers=False,
                        showmeans=True, showfliers=False,
                        # patch_artist=True, notch=True,
                        meanprops=dict(marker="+", markeredgecolor="black"),
                        # medianprops=dict(color="magenta"),
                        boxprops=dict(linewidth=2, color=color),
                        whiskerprops=dict(linewidth=2, color=color),
                        capprops=dict(linewidth=2, color=color),
                        # meanprops=dict(marker="+", markeredgecolor="tab:cyan"),
                        # boxprops=dict(facecolor=c, color=c),
                        # capprops=dict(color=c),
                        # whiskerprops=dict(color=c),
                        # flierprops=dict(color=c, markeredgecolor=c),
                        # medianprops=dict(color=c),
                        widths=1.0)
        #ax.set_xticks(heights[::4], [str(h) for h in heights[::4]])
        a = heights[::4]
        ax.set_xticks(a)
        ax.set_xticklabels(a)
        #ax.set_xticks(ax.get_xticks()[::4], ax.get_xticklabels()[::4])
        #ax.set_xticks([0, 10, 20], ["a", "b", "c"])
        # ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        # ax.locator_params(axis='x', nbins=10)
        # bp = ax.boxplot(collection[i], positions=[0, spacing])
        # ax.set_xlim(-0.05, 0.45)
        # for i in range(len(bp["boxes"])):
        #    c = colors[i]
        #    bp["boxes"][i].set_color(c)
        #    #bp["fliers"][i].set_color(c)
        #    bp["whiskers"][2 * i].set_color(c)
        #    bp["whiskers"][2 * i + 1].set_color(c)
        #    bp["caps"][2 * i].set_color(c)
        #    bp["caps"][2 * i + 1].set_color(c)


def box_plot(data_ref, data_par, categories):
    results_ref = {}
    results_par = {}
    for k in categories.keys():
        results_ref[k] = {}
        results_par[k] = {}
        for method, d in data_ref.items():
            results_ref[k][method] = []
            for v in d.values():
                if v[k] is not None:
                    results_ref[k][method].append(v[k])
            results_ref[k][method] = np.array(results_ref[k][method]).astype(np.float32)
        for method, d in data_par.items():
            results_par[k][method] = []
            for v in d.values():
                if v[k] is not None:
                    results_par[k][method].append(v[k])
            results_par[k][method] = np.array(results_par[k][method]).astype(np.float32)

    plt.rc('font', size=15)
    plt.rc('legend', fontsize=17)
    # plt.figure(figsize=(18, 4))
    plt.figure(figsize=(9, 7))
    for i, (c, cv) in enumerate(categories.items()):
        ax = plt.subplot(1, 2, 1 + i)
        if cv.log:
            ax.set_yscale('log')
        ax.set_title(cv.title, rotation=0, ha="center", x=0.60)
        ax.set_xlabel("Table height [cm]")
        datapoints_ref = [results_ref[c][k] * cv.scale for k in planners_ref_names]
        datapoints_par = [results_par[c][k] * cv.scale for k in planners_par_names]
        plot_datapoints(ax, c, datapoints_ref, "b", "o")
        plot_datapoints(ax, c, datapoints_par, "r", "x")
        if c in ["scored", "hit"]:
            ax.plot([10, 10], [0, 100], 'g--')
            ax.plot([20, 20], [0, 100], 'g--')
        else:
            ax.plot([10, 10], [0, 70], 'g--')
            ax.plot([20, 20], [0, 70], 'g--')


        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    plt.subplots_adjust(hspace=0.3, bottom=0.2)
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


results_ref = {}
results_par = {}
for path in glob("../results/hitting_exp/*7jc"):
    data = read_results(path)
    name = path.split("/")[-1]
    if name in planners_ref_names_map.keys():
        results_ref[name] = data
    if name in planners_par_names_map.keys():
        results_par[name] = data

description = namedtuple("Description", "title scale log")
categories = {
    "scored": description("Score ratio [%]", 100., False),
    "actual_z_error": description("Z-axis error [mm⋅s]", 1000., False),
}
box_plot(results_ref, results_par, categories)