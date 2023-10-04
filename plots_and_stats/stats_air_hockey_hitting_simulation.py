import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import namedtuple, OrderedDict
import scipy.stats as ss
import scikit_posthocs as sp
from statsmodels.stats.contingency_tables import cochrans_q, mcnemar

root_dir = os.path.dirname(__file__)
package_dir = os.path.dirname(root_dir)

planners_names_map = dict(ours_aramis_="CNP-B (ours)", sst="SST [41]", mpcmpnet="MPC-MPNet [54]", iros_="AQP [8]",
                          nlopt="TrajOpt [25]", cbirrt="CBiRRT [11]")
colors = OrderedDict(ours_aramis_='tab:blue', iros_='tab:orange', nlopt='tab:green', sst='tab:red',
                     mpcmpnet='tab:purple',
                     cbirrt='tab:brown')

def mean(r, k, abs=True):
    s = None
    if abs:
        s = [np.abs(v[k]) for _, v in r.items() if k in v and v[k] is not None]
    else:
        s = [v[k] for _, v in r.items() if k in v and v[k] is not None]
    return np.mean(s)


def stats(data, categories):
    extract = lambda x, y: [v[y] for _, v in x.items() if y in v]
    results = {}
    results_with_keys = {}
    for k in categories.keys():
        results[k] = {}
        results_with_keys[k] = {}
        for method, d in data.items():
            results[k][method] = []
            results_with_keys[k][method] = {}
            for dk, v in d.items():
                # if k in v and v[k] is not None and v["hit"]:
                if k in v and v[k] is not None:
                    results[k][method].append(v[k])
                    results_with_keys[k][method][dk] = v[k]
            results[k][method] = np.array(results[k][method]).astype(np.float32)

    methods = list(results['scored'].keys())
    scored = np.stack([results['scored'][m] for m in methods], axis=1)

    ours_ok = np.sum(scored[:, 0])
    ours_err = scored.shape[0] - ours_ok
    iros_ok = np.sum(scored[:, 3])
    iros_err = scored.shape[0] - iros_ok
    contingency_valid_ours_iros = np.array([[ours_ok, ours_err], [iros_err, iros_ok]])
    nlopt_ok = np.sum(scored[:, 4])
    nlopt_err = scored.shape[0] - nlopt_ok
    contingency_valid_ours_nlopt = np.array([[ours_ok, ours_err], [nlopt_err, nlopt_ok]])

    print("SCORED OURS VS AQP:")
    print(mcnemar(contingency_valid_ours_iros, exact=False, correction=False))
    print("SCORED OURS VS TrajOpt:")
    print(mcnemar(contingency_valid_ours_nlopt, exact=False, correction=False))

    print("PLANNING TIME OURS VS AQP:")
    print(ss.wilcoxon(results['planning_time']['ours_aramis_'], results['planning_time']['iros_'], alternative="less"))
    print(ss.ttest_rel(results['planning_time']['ours_aramis_'], results['planning_time']['iros_'], alternative="less"))


    print("PUCK VELOCITY OURS VS AQP:")
    print(ss.wilcoxon(results['puck_velocity_magnitude']['ours_aramis_'], results['puck_velocity_magnitude']['iros_'], alternative='greater'))
    print(ss.ttest_rel(results['puck_velocity_magnitude']['ours_aramis_'], results['puck_velocity_magnitude']['iros_'], alternative='greater'))

    print("ACTUAL HITTING TIME:")
    key = 'actual_hitting_time'
    ours_l2048_keys = results_with_keys[key]['ours_aramis_'].keys()
    iros_keys = results_with_keys[key]['iros_'].keys()
    mpcmpnet_keys = results_with_keys[key]['mpcmpnet'].keys()

    ours_iros_keys = list(set(ours_l2048_keys) & set(iros_keys))
    ours_chosen = [results_with_keys[key]['ours_aramis_'][k] for k in ours_iros_keys]
    iros_chosen = [results_with_keys[key]['iros_'][k] for k in ours_iros_keys]
    print("OURS VS AQP")
    print(ss.wilcoxon(ours_chosen, iros_chosen, alternative='less'))
    print(ss.ttest_rel(ours_chosen, iros_chosen, alternative='less'))

    ours_mpcmpnet_keys = list(set(ours_l2048_keys) & set(mpcmpnet_keys))
    ours_chosen = [results_with_keys[key]['ours_aramis_'][k] for k in ours_mpcmpnet_keys]
    mpcmpnet_chosen = [results_with_keys[key]['mpcmpnet'][k] for k in ours_mpcmpnet_keys]
    #actual_hitting_time_stats_ours_mpcmpnet = ss.wilcoxon(ours_chosen, mpcmpnet_chosen)
    print("OURS VS MPNET")
    print(ss.wilcoxon(ours_chosen, mpcmpnet_chosen, alternative='greater'))
    print(ss.ttest_rel(ours_chosen, mpcmpnet_chosen, alternative='greater'))


    print("PLANNED Z ERROR")
    key = 'planned_z_error'
    ours_l2048_keys = results_with_keys[key]['ours_aramis_'].keys()
    nlopt_keys = results_with_keys[key]['nlopt'].keys()
    mpcmpnet_keys = results_with_keys[key]['mpcmpnet'].keys()

    ours_nlopt_keys = list(set(ours_l2048_keys) & set(nlopt_keys))
    ours_chosen = [results_with_keys[key]['ours_aramis_'][k] for k in ours_nlopt_keys]
    nlopt_chosen = [results_with_keys[key]['nlopt'][k] for k in ours_nlopt_keys]
    print("OURS VS TrajOpt")
    print(ss.wilcoxon(ours_chosen, nlopt_chosen, alternative='less'))
    print(ss.ttest_rel(ours_chosen, nlopt_chosen, alternative='less'))

    ours_mpcmpnet_keys = list(set(ours_l2048_keys) & set(mpcmpnet_keys))
    ours_chosen = [results_with_keys[key]['ours_aramis_'][k] for k in ours_mpcmpnet_keys]
    mpcmpnet_chosen = [results_with_keys[key]['mpcmpnet'][k] for k in ours_mpcmpnet_keys]
    print("OURS VS MPNET")
    print(ss.wilcoxon(ours_chosen, mpcmpnet_chosen, alternative='greater'))
    print(ss.ttest_rel(ours_chosen, mpcmpnet_chosen, alternative='less'))


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
    "hit": description("Hit ratio [%]", 100., False),
    "planning_time": description("Planning time [ms]", 1., True),
    "puck_velocity_magnitude": description("Puck velocity [m/s]", 1., False),
    "actual_hitting_time": description("Hitting time [s]", 1., False),
    "actual_z_error": description("Z-axis error [mm⋅s]", 1000., True),
    "planned_z_error": description("Z-axis error [mm⋅s]", 1000., False),
}
summary = {k: {} for k, v in categories.items()}
for name, data in results.items():
    for cat_k, cat_v in categories.items():
        summary[cat_k][name] = mean(data, cat_k)

stats(results, categories)