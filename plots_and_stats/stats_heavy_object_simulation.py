import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import namedtuple
import matplotlib
from scipy.stats import wilcoxon, ttest_rel
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar

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
colors_list = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray",
               "tab:olive", ]  # "tab:cyan"]


def mean(r, k, filter_finished):
    if filter_finished:
        s = [v[k] for _, v in r.items() if (k in v and v[k] is not None and v["finished"])]
    else:
        s = [v[k] for _, v in r.items() if k in v and v[k] is not None]
    return np.mean(s)


def stats(data, categories):
    results = {}
    results_with_keys = {}
    for k in categories.keys():
        results[k] = {}
        results_with_keys[k] = {}
        for method, d in data.items():
            results[k][method] = []
            results_with_keys[k][method] = {}
            for dk, v in d.items():
                if k in v.keys() and v[k] is not None:
                    results[k][method].append(v[k])
                    results_with_keys[k][method][dk] = v[k]
            results[k][method] = np.array(results[k][method]).astype(np.float32)
    # positions = np.reshape(np.array([[i, i + spacing] for i in range(int(len(data) / 2.))]), -1)
    for k, v in results["alpha_beta"].items():
        results["alpha_beta"][k] = results["alpha_beta"][k][v != 0]

    methods = list(planners_names_order.keys())
    valid = np.stack([results['valid'][m] for m in methods], axis=1)
    finished = np.stack([results['finished'][m] for m in methods], axis=1)

    print("SUCCESS W/ CONSTRAINTS:")
    ours_ok = np.sum(valid[:, 0])
    ours_err = valid.shape[0] - ours_ok
    cbirrt_ok = np.sum(valid[:, 1])
    cbirrt_err = valid.shape[0] - cbirrt_ok
    contingency_valid_ours_cbirrt = np.array([[ours_ok, ours_err], [cbirrt_err, cbirrt_ok]])
    print(mcnemar(contingency_valid_ours_cbirrt, exact=False, correction=False))

    print("SUCCESS W/O CONSTRAINTS:")
    ours_ok = np.sum(finished[:, 0])
    ours_err = finished.shape[0] - ours_ok
    cbirrt_ok = np.sum(finished[:, 1])
    cbirrt_err = finished.shape[0] - cbirrt_ok
    contingency_finished_ours_cbirrt = np.array([[ours_ok, ours_err], [cbirrt_err, cbirrt_ok]])
    print(mcnemar(contingency_finished_ours_cbirrt, exact=False, correction=False))

    print("PLANNING TIME OURS VS CBIRRT:")
    print(wilcoxon(results['planning_time']['ours_l2048'], results['planning_time']['cbirrt'], alternative='less'))
    print(ttest_rel(results['planning_time']['ours_l2048'], results['planning_time']['cbirrt'], alternative='less'))

    print("MOTION TIME:")
    ours_l2048_keys = results_with_keys['motion_time']['ours_l2048'].keys()
    nlopt_keys = results_with_keys['motion_time']['nlopt'].keys()
    mpcmpnet_keys = results_with_keys['motion_time']['mpcmpnet'].keys()

    ours_nlopt_keys = list(set(ours_l2048_keys) & set(nlopt_keys))
    ours_chosen = [results_with_keys['motion_time']['ours_l2048'][k] for k in ours_nlopt_keys]
    nlopt_chosen = [results_with_keys['motion_time']['nlopt'][k] for k in ours_nlopt_keys]
    print("OURS VS TrajOpt")
    print(wilcoxon(ours_chosen, nlopt_chosen, alternative="less"))
    print(ttest_rel(ours_chosen, nlopt_chosen, alternative="less"))

    ours_mpcmpnet_keys = list(set(ours_l2048_keys) & set(mpcmpnet_keys))
    ours_chosen = [results_with_keys['motion_time']['ours_l2048'][k] for k in ours_mpcmpnet_keys]
    mpcmpnet_chosen = [results_with_keys['motion_time']['mpcmpnet'][k] for k in ours_mpcmpnet_keys]
    print("OURS VS MPCMPNet")
    print(wilcoxon(ours_chosen, mpcmpnet_chosen, alternative="less"))
    print(ttest_rel(ours_chosen, mpcmpnet_chosen, alternative="less"))

    print("VERTICAL ERROR OURS VS TrajOpt:")
    ours_l2048_keys = results_with_keys['alpha_beta']['ours_l2048'].keys()
    nlopt_keys = results_with_keys['alpha_beta']['nlopt'].keys()
    ours_nlopt_keys = list(set(ours_l2048_keys) & set(nlopt_keys))
    ours_chosen = [results_with_keys['alpha_beta']['ours_l2048'][k] for k in ours_nlopt_keys]
    nlopt_chosen = [results_with_keys['alpha_beta']['nlopt'][k] for k in ours_nlopt_keys]
    print(wilcoxon(ours_chosen, nlopt_chosen, alternative="less"))
    print(ttest_rel(ours_chosen, nlopt_chosen, alternative="less"))


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

stats(results, categories)
