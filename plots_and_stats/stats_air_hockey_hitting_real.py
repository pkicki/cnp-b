import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from collections import namedtuple
import scipy.stats as ss
from statsmodels.stats.contingency_tables import mcnemar

root_dir = os.path.dirname(__file__)
print(root_dir)
package_dir = os.path.dirname(root_dir)
baseline_path = "../results/hitting_exp/baseline_opt_lp/"
ours_path = "../results/hitting_exp/ours_opt_lp/"
print(ours_path)

def stats(ours, baseline):
    good_indices = [k for k, v in baseline.items() if v['hit']]

    scored_ours = np.array([ours[k]['scored'] for k in good_indices])
    scored_baseline = np.array([baseline[k]['scored'] for k in good_indices])

    hitting_time_ours = np.array([ours[k]['actual_hitting_time'] for k in good_indices])
    hitting_time_baseline = np.array([baseline[k]['actual_hitting_time'] for k in good_indices])

    puck_velocity_ours = np.array([ours[k]['puck_velocity_magnitude'] for k in good_indices])
    puck_velocity_baseline = np.array([baseline[k]['puck_velocity_magnitude'] for k in good_indices])

    planned_z_error_ours = np.array([ours[k]['planned_z_error'] for k in good_indices])
    planned_z_error_baseline = np.array([baseline[k]['planned_z_error'] for k in good_indices])

    actual_z_error_ours = np.array([ours[k]['actual_z_error'] for k in good_indices])
    actual_z_error_baseline = np.array([baseline[k]['actual_z_error'] for k in good_indices])

    joint_trajectory_error_ours = np.array([ours[k]['joint_trajectory_error'] for k in good_indices])
    joint_trajectory_error_baseline = np.array([baseline[k]['joint_trajectory_error'] for k in good_indices])


    all = len(good_indices)
    ours_ok = np.sum(scored_ours)
    ours_err = all - ours_ok
    baseline_ok = np.sum(scored_baseline)
    baseline_err = all - baseline_ok
    contingency_valid = np.array([[ours_ok, ours_err], [baseline_err, baseline_ok]])

    print("SCORING RATIOS:")
    print(mcnemar(contingency_valid, exact=False, correction=False))

    print("HITTING TIME:")
    print(ss.wilcoxon(hitting_time_baseline, hitting_time_ours, alternative='greater'))
    print(ss.ttest_rel(hitting_time_baseline, hitting_time_ours, alternative='greater'))

    print("PUCK VELOCITY:")
    print(ss.wilcoxon(puck_velocity_baseline, puck_velocity_ours, alternative='less'))
    print(ss.ttest_rel(puck_velocity_baseline, puck_velocity_ours, alternative='less'))

    print("PLANNED Z ERROR:")
    print(ss.wilcoxon(planned_z_error_baseline, planned_z_error_ours, alternative='greater'))
    print(ss.ttest_rel(planned_z_error_baseline, planned_z_error_ours, alternative='greater'))

    print("ACTUAL Z ERROR:")
    print(ss.wilcoxon(actual_z_error_baseline, actual_z_error_ours, alternative='greater'))
    print(ss.ttest_rel(actual_z_error_baseline, actual_z_error_ours, alternative='greater'))

    print("JOINT TRAJECTORY ERROR:")
    print(ss.wilcoxon(joint_trajectory_error_baseline, joint_trajectory_error_ours, alternative='greater'))
    print(ss.ttest_rel(joint_trajectory_error_baseline, joint_trajectory_error_ours, alternative='greater'))


def mean(r, k, abs=True):
    s = [v[k] for _, v in r.items() if k in v] if not abs else [np.abs(v[k]) for _, v in r.items() if k in v]
    return np.mean(s)


def read_results(path):
    results = {}
    for p in glob(path + "*.res"):
        with open(p, 'rb') as fh:
            d = pickle.load(fh)
            results[p[-6:-4]] = d
    return results


ours = read_results(ours_path)
baseline = read_results(baseline_path)
stats(ours, baseline)