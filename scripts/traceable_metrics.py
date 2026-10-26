#!/usr/bin/env python3
"""Compute every entry of Tables 4 and 5 of the paper from the released logs.

Every reinforcement learning result in the paper is computed by this script
from the TensorBoard event files in logs/. Nothing else is needed.

    unzip logs/logs.zip -d logs/
    python scripts/traceable_metrics.py --root logs/logs --out stats

Aggregation (Section 5.2 of the paper)
--------------------------------------
1. Within each run, a metric is reduced to ONE run-level value.
2. Across runs, the run-level values are summarised as mean +/- SD (ddof=1),
   with the number of contributing runs reported when it is fewer than five.

Metrics
-------
return_early / return_late
    Mean of rollout/ep_rew_mean over the absolute step windows
    [100k, 200k) and [800k, 900k). A run contributes to a window only if it
    reaches the end of that window. The early window is reached by all twenty
    runs, so it is unaffected by the unequal training budgets.
peak
    Maximum unsmoothed rollout/ep_rew_mean before 900k steps.
laps_late
    Mean of episode/laps_completed over [800k, 900k).
steps_to_1000
    First logged step at which rollout/ep_rew_mean >= 1000.
lap statistics
    time/lap_time events before 900k steps, excluding events < 15 s (timing
    artefacts, the same threshold as make_figures.py). Per run: best, median,
    interquartile range.

Table 5 (sensitivity)
---------------------
Within each configuration, three runs share a logging structure (Section 7.3
of the paper; listed in SHARED below). Table 5 averages those three into one
run before aggregation, leaving at most three effective runs.

Outputs (in --out)
------------------
per_run_traceable.csv   one row per run, every run-level metric
per_run_laps.csv        per-run lap statistics
laps_traceable.csv      every lap event used (config, run, step, lap_time)
table4_summary.csv      the across-run summary shown in Table 4
table5_sensitivity.csv  the collapsed-trio summary shown in Table 5

Part of the code and data release for Haja et al., Array, 2026. Licensed MIT.
"""

import argparse
import glob
import importlib.util
import os

import numpy as np
import pandas as pd

CONDITIONS = {
    "Config 1": "cnnPolicy-minimonacoTrack",
    "Config 2": "car_srl_ae_32_no_speed_minimonacoTrack",
    "Config 3": "car_srl_ae_32_with_speed_minimonacoTrack",
    "Config 4": "car_srl_ae_64_with_speed_minimonacoTrack",
}
SHARED = {
    "Config 1": {"SAC_1", "SAC_4", "SAC_5"},
    "Config 2": {"SAC_3", "SAC_4", "SAC_5"},
    "Config 3": {"SAC_2", "SAC_4", "SAC_5"},
    "Config 4": {"SAC_1", "SAC_4", "SAC_5"},
}
WINDOWS = {"early": (100_000, 200_000), "late": (800_000, 900_000)}
HORIZON = 900_000
LAP_MIN = 15.0
THRESHOLD = 1000.0


def load_reader():
    """Reuse the event-file parser in make_figures.py (no TensorFlow needed)."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("mf", os.path.join(here, "make_figures.py"))
    mf = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mf)
    return mf.read_scalars


def mean_sd(values):
    v = np.array([x for x in values if pd.notna(x)], dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, 0
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else np.nan), len(v)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="logs/logs", help="folder holding the four condition folders")
    ap.add_argument("--out", default="stats")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    read_scalars = load_reader()

    runs, laps = [], []
    for cond, folder in CONDITIONS.items():
        dirs = [d for d in sorted(glob.glob(os.path.join(args.root, "**", folder, "**", "SAC_*"),
                                            recursive=True)) if os.path.isdir(d)]
        if not dirs:
            raise SystemExit("no runs found for %s under %s" % (cond, args.root))
        for d in dirs:
            run = os.path.basename(d)
            df = pd.concat([read_scalars(e) for e in glob.glob(os.path.join(d, "events.out.tfevents.*"))],
                           ignore_index=True)
            rew = df[df.tag == "rollout/ep_rew_mean"].sort_values("step")
            lc = df[df.tag == "episode/laps_completed"].sort_values("step")
            lap = df[df.tag == "time/lap_time"].sort_values("step")
            last = int(rew.step.max())
            row = dict(config=cond, run=run, shared=run in SHARED[cond], last_step=last)
            for name, (a, b) in WINDOWS.items():
                reached = last >= b
                seg = rew[(rew.step >= a) & (rew.step < b)]
                lseg = lc[(lc.step >= a) & (lc.step < b)]
                row["return_" + name] = float(seg.value.mean()) if reached and len(seg) else np.nan
                row["laps_" + name] = float(lseg.value.mean()) if reached and len(lseg) else np.nan
            row["peak"] = float(rew[rew.step < HORIZON].value.max())
            hit = rew[rew.value >= THRESHOLD]
            row["steps_to_1000"] = int(hit.step.iloc[0]) if len(hit) else np.nan
            runs.append(row)
            for t, s in zip(lap.step, lap.value):
                if s >= LAP_MIN and t < HORIZON:
                    laps.append((cond, run, int(t), float(s)))

    R = pd.DataFrame(runs).sort_values(["config", "run"])
    L = pd.DataFrame(laps, columns=["config", "run", "step", "lap_time"])
    P = (L.groupby(["config", "run"]).lap_time
          .agg(n_laps="count", best="min", median="median",
               iqr=lambda s: s.quantile(0.75) - s.quantile(0.25))
          .reset_index())
    R.to_csv(os.path.join(args.out, "per_run_traceable.csv"), index=False)
    L.to_csv(os.path.join(args.out, "laps_traceable.csv"), index=False)
    P.to_csv(os.path.join(args.out, "per_run_laps.csv"), index=False)

    t4 = []
    for cond in CONDITIONS:
        r, p = R[R.config == cond], P[P.config == cond]
        row = {"config": cond}
        for col in ["return_early", "return_late", "peak", "laps_late", "steps_to_1000"]:
            m, s, n = mean_sd(r[col])
            row[col + "_mean"], row[col + "_sd"], row[col + "_n"] = m, s, n
        if cond != "Config 1":
            row["best_lap"] = p.best.min()
            for col in ["median", "iqr"]:
                m, s, n = mean_sd(p[col])
                row["lap_%s_mean" % col], row["lap_%s_sd" % col] = m, s
        t4.append(row)
    T4 = pd.DataFrame(t4)
    T4.to_csv(os.path.join(args.out, "table4_summary.csv"), index=False)

    t5 = []
    for cond in CONDITIONS:
        r = R[R.config == cond]
        row = {"config": cond}
        for w in WINDOWS:
            col = "return_" + w
            ind = r[~r.shared][col].dropna().tolist()
            trio = r[r.shared][col].dropna()
            v = np.array(ind + ([trio.mean()] if len(trio) else []))
            row[w + "_mean"], row[w + "_min"], row[w + "_max"], row[w + "_n_effective"] = (
                v.mean(), v.min(), v.max(), len(v))
        t5.append(row)
    T5 = pd.DataFrame(t5)
    T5.to_csv(os.path.join(args.out, "table5_sensitivity.csv"), index=False)

    pd.set_option("display.width", 200)
    print("Table 4\n" + T4.round(1).to_string(index=False))
    print("\nTable 5\n" + T5.round(0).to_string(index=False))


if __name__ == "__main__":
    main()
