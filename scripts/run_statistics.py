#!/usr/bin/env python3
"""Per-run metrics and inferential tests across the five runs of each condition.

Answers the request for a small-sample test on the run-level results. Reads the
TensorBoard event files directly -- no TensorFlow required -- reusing the parser
in make_figures.py so the numbers here and the numbers in the figures come from
exactly the same source.

    python scripts/run_statistics.py --root logs --out stats

Outputs
-------
    per_run_metrics.csv   one row per run: asymptotic return, peak, last step
    pairwise_tests.csv    Welch t, Mann-Whitney U, Cliff's delta per comparison

Definition of asymptotic return: the mean of rollout/ep_rew_mean over the
absolute step range [BUDGET - ASYM_WINDOW, BUDGET), identical for every run and
every configuration. This is a run-level summary, so each configuration
contributes exactly five numbers to every test -- which is what makes a
between-run test legitimate. Averaging within a run first is essential: treating
individual episodes as samples would inflate n by pooling correlated data from
the same policy.

CAVEAT. The tests below assume the five runs of a configuration are independent
replicates. The released event files show that three of the five runs in each
configuration share an identical episode-boundary structure, and coincide
exactly on all metrics over part of the final phase of training (Section 7.3 of
the paper). That assumption therefore does not hold, which is why the paper
reports run-to-run spread descriptively and does not report these tests. This
script is retained so the per-run metrics can be recomputed, not to support an
inferential claim.

With n = 5 per group the smallest attainable two-sided Mann-Whitney p is 0.0079,
so that value indicates complete separation rather than a marginal result. Read
it alongside Cliff's delta, which reports the fraction of cross-group pairs in
which the first group wins.

Part of the code and data release for Haja et al., Array (under review), 2026.
Licensed MIT.
"""

import argparse
import glob
import importlib.util
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

# One window definition, shared with make_figures.py: an ABSOLUTE step range,
# not each run's own final steps. Using each run's own tail put the window at
# different absolute ranges across configurations (roughly 1.00-1.10e6 for
# Config 3 against 0.82-0.92e6 for Config 1), which made the configurations
# non-comparable. Runs that never reach BUDGET - ASYM_WINDOW contribute no
# value and are reported as NaN rather than silently summarised over a
# different part of training.
BUDGET      = 900_000
ASYM_WINDOW = 100_000
TAG = "rollout/ep_rew_mean"

CONDITIONS = {
    "Config 1": "cnnPolicy-minimonacoTrack",
    "Config 2": "car_srl_ae_32_no_speed_minimonacoTrack",
    "Config 3": "car_srl_ae_32_with_speed_minimonacoTrack",
    "Config 4": "car_srl_ae_64_with_speed_minimonacoTrack",
}


def load_reader():
    """Reuse the event parser from make_figures.py."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "mf", os.path.join(here, "make_figures.py"))
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    return mf.read_scalars


def bootstrap_ci(x, n_boot=20000, seed=0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, (n_boot, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def cliffs_delta(x, y):
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    return float((np.sum(x > y) - np.sum(x < y)) / (x.size * y.size))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="logs")
    ap.add_argument("--out", default=".")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    read_scalars = load_reader()

    rows = []
    for cond, folder in CONDITIONS.items():
        runs = [d for d in sorted(
            glob.glob(os.path.join(args.root, "**", folder, "**", "SAC_*"),
                      recursive=True)) if os.path.isdir(d)]
        if not runs:
            print("  no runs found for %s" % cond)
        for run in runs:
            events = glob.glob(os.path.join(run, "events.out.tfevents.*"))
            df = pd.concat([read_scalars(e) for e in events], ignore_index=True)
            s = df[df.tag == TAG].sort_values("step")
            if s.empty:
                continue
            last = int(s.step.max())
            rows.append(dict(
                config=cond, run=os.path.basename(run), last_step=last,
                asymptotic_return=float(
                    s[(s.step >= BUDGET - ASYM_WINDOW) & (s.step < BUDGET)].value.mean()),
                peak_return=float(s.value.max())))

    per_run = pd.DataFrame(rows).sort_values(["config", "run"])
    per_run.to_csv(os.path.join(args.out, "per_run_metrics.csv"), index=False)
    print(per_run.to_string(index=False), "\n")

    groups = {c: g.asymptotic_return.values for c, g in per_run.groupby("config")}
    print("%-10s %3s %10s %10s   %s" % ("config", "n", "mean", "sd", "95% bootstrap CI"))
    for c, v in groups.items():
        lo, hi = bootstrap_ci(v)
        print("%-10s %3d %10.1f %10.1f   [%.1f, %.1f]"
              % (c, len(v), v.mean(), v.std(ddof=1), lo, hi))

    out = []
    for a, b in itertools.combinations(sorted(groups), 2):
        x, y = groups[a], groups[b]
        t, p_t = stats.ttest_ind(x, y, equal_var=False)
        u, p_u = stats.mannwhitneyu(x, y, alternative="two-sided")
        out.append(dict(group_a=a, group_b=b, mean_a=x.mean(), mean_b=y.mean(),
                        welch_t=t, welch_p=p_t, mwu_u=u, mwu_p=p_u,
                        cliffs_delta=cliffs_delta(x, y)))
    tests = pd.DataFrame(out)
    tests.to_csv(os.path.join(args.out, "pairwise_tests.csv"), index=False)
    print("\n" + tests.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
