#!/usr/bin/env python3
"""
Figure generation for the vision-based autonomous racing SAC ablations
(state-representation variants: latent size, speed input, autoencoder budget).

Reads TensorBoard event files for the four experimental conditions and emits
every figure used in the paper, plus a CSV of all per-run aggregate metrics.

Usage
-----
    pip install matplotlib pandas numpy scipy
    (no tensorflow / tbparse needed - event files are parsed directly)
    python make_figures.py --root /path/to/logs --out ./figures

Expected directory layout under --root (folder names configurable below).
Event files are located recursively, so the intermediate environment folder
and the per-seed SAC_* folders are picked up automatically:

    car_srl_ae_32_no_speed_minimonacoTrack/
        donkey-minimonaco-track-v0/SAC_1/events.out.tfevents.*
        donkey-minimonaco-track-v0/SAC_2/events.out.tfevents.*
        donkey-minimonaco-track-v0/SAC_3/events.out.tfevents.*
    car_srl_ae_32_with_speed_minimonacoTrack/            ... same layout
    car_srl_ae_64_with_speed_minimonacoTrack/            ... same layout

One run = one SAC_* directory. If such a directory holds several event files
(e.g. a resumed run), they are concatenated into a single run.

Aggregation follows Agarwal et al. (NeurIPS 2021): the interquartile mean
(25% trimmed mean) across runs, with the min-max envelope shown as a band and
individual runs drawn faintly behind. Curves are resampled onto a common step
grid before aggregation so that runs with different logging cadences combine
correctly.
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
import numpy as np
warnings.filterwarnings("ignore", category=RuntimeWarning)
import pandas as pd
from matplotlib.lines import Line2D
from scipy.stats import trim_mean
# --------------------------------------------------------------------------
# TensorBoard event reading (no tensorflow / tbparse / protobuf required)
#
# tbparse imports tensorflow, which raises on environments where an old
# tensorflow meets protobuf >= 4 ("Descriptors cannot be created directly").
# Scalars are all this script needs, so the TFRecord container and the handful
# of protobuf fields involved are parsed directly below. Verified to produce
# byte-identical values to tbparse on these logs.
# --------------------------------------------------------------------------
import struct


def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s
        if not c & 0x80:
            return r, i
        s += 7


def _pb_fields(b):
    """Yield (field_number, wire_type, payload) for one protobuf message."""
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        fn, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 1:
            v, i = b[i:i + 8], i + 8
        elif wt == 2:
            ln, i = _varint(b, i)
            v, i = b[i:i + ln], i + ln
        elif wt == 5:
            v, i = b[i:i + 4], i + 4
        else:
            return
        yield fn, wt, v


def _tf_records(path):
    """Yield the payload of each TFRecord (length, crc, data, crc)."""
    with open(path, "rb") as f:
        buf = f.read()
    i, n = 0, len(buf)
    while i + 12 <= n:
        (ln,) = struct.unpack("<Q", buf[i:i + 8])
        i += 12                       # 8-byte length + 4-byte length crc
        if i + ln + 4 > n:
            return                    # truncated tail (run killed mid-write)
        yield buf[i:i + ln]
        i += ln + 4                   # payload + data crc


def _tensor_scalar(payload):
    """Pull a single float out of a TensorProto (new-style scalar logging)."""
    dtype, content, vals = None, None, []
    for fn, wt, v in _pb_fields(payload):
        if fn == 1 and wt == 0:
            dtype = v                                    # DT_FLOAT=1, DT_DOUBLE=2
        elif fn == 4 and wt == 2:
            content = v                                  # packed tensor_content
        elif fn == 5:
            vals += (list(struct.unpack(f"<{len(v)//4}f", v)) if wt == 2
                     else [struct.unpack("<f", v)[0]])
        elif fn == 6:
            vals += (list(struct.unpack(f"<{len(v)//8}d", v)) if wt == 2
                     else [struct.unpack("<d", v)[0]])
    if vals:
        return float(vals[0])
    if content:
        fmt = "<d" if dtype == 2 else "<f"
        w = struct.calcsize(fmt)
        if len(content) >= w:
            return float(struct.unpack(fmt, content[:w])[0])
    return None


def read_scalars(path):
    """Read one event file into a long-format DataFrame (tag, step, value)."""
    rows = []
    for rec in _tf_records(path):
        step, summary = 0, None
        for fn, wt, v in _pb_fields(rec):
            if fn == 2 and wt == 0:
                step = v                                 # Event.step
            elif fn == 5 and wt == 2:
                summary = v                              # Event.summary
        if summary is None:
            continue
        for fn, wt, v in _pb_fields(summary):
            if fn != 1 or wt != 2:
                continue                                 # Summary.value (repeated)
            tag, val = None, None
            for sfn, swt, sv in _pb_fields(v):
                if sfn == 1 and swt == 2:
                    tag = sv.decode("utf-8", "replace")
                elif sfn == 2 and swt == 5:
                    val = struct.unpack("<f", sv)[0]     # simple_value
                elif sfn == 8 and swt == 2:
                    val = _tensor_scalar(sv)             # tensor
            if tag is not None and val is not None:
                rows.append((tag, step, val))
    return pd.DataFrame(rows, columns=["tag", "step", "value"])

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

# condition key -> folder name directly under --root
CONDITIONS = {
    "CNN":          "cnnPolicy-minimonacoTrack",
    "AE32_nospeed": "car_srl_ae_32_no_speed_minimonacoTrack",
    "AE32_speed":   "car_srl_ae_32_with_speed_minimonacoTrack",
    "AE64_speed":   "car_srl_ae_64_with_speed_minimonacoTrack",
}

# Event files sit two levels below the condition folder
# (<env>/SAC_<seed>/), so the search is recursive.
EVENT_GLOB = os.path.join("**", "events.out.tfevents.*")

LABELS = {
    "CNN":          "End-to-end CNN policy",
    "AE32_nospeed": "AE-32, no speed",
    "AE32_speed":   "AE-32 + speed",
    "AE64_speed":   "AE-64 + speed",
}

# Colour identifies the condition; line style encodes the state input
# (dashed = no speed in the observation, solid = speed appended,
#  dash-dot = speed + longer autoencoder training).
COLORS = {
    "CNN":          "#6b4c9a",
    "AE32_nospeed": "#7a7a7a",
    "AE32_speed":   "#1f5fa9",
    "AE64_speed":   "#e07b39",
}
STYLES = {
    "CNN":          ":",
    "AE32_nospeed": "--",
    "AE32_speed":   "-",
    "AE64_speed":   "-",
}

# Each comparison isolates a single factor: reference condition, variant
# condition, output file stem, and figure title.
PAIRS = {
    "speed":  ("AE32_nospeed", "AE32_speed", "fig3_ablation_speed",
               "Speed in the observation: AE-32 without vs. with speed"),
    "latent": ("AE32_speed", "AE64_speed", "fig4_ablation_latent_dim",
               "Latent size: AE-32 vs. AE-64 (both + speed)"),
    "srl":    ("CNN", "AE32_speed", "fig5_srl_vs_cnn",
               "State representation: end-to-end CNN vs. AE-32 + speed"),
}

# EMA weight on each new sample, for curve display only (metrics use raw data).
# Lower = smoother. Equivalent to a TensorBoard smoothing slider of 1 - value,
# so 0.20 here corresponds to TensorBoard's 0.80.
SMOOTH_ALPHA = 0.20

BUDGET = 900_000          # common truncation point for all *metrics*
PLOT_XMAX = 1_000_000     # x-extent for the return curve (display only;
                          # metrics stay on BUDGET so all runs contribute)
ASYM_WINDOW = 100_000     # asymptotic window = final 100k steps
JUMP_WINDOW = 10_000      # early-learning window = first 10k steps
THRESHOLD = 500           # time-to-threshold: mean episode length >= 500
LAP_MIN_VALID = 15.0      # lap times below this are detection artifacts

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 300, "font.size": 9,
    "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "savefig.bbox": "tight", "figure.autolayout": False,
})


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_runs(root):
    """Return {condition: [DataFrame per run]} of long-format scalars."""
    data = {}
    for cond, folder in CONDITIONS.items():
        pattern = os.path.join(root, folder, EVENT_GLOB)
        files = sorted(glob.glob(pattern, recursive=True))
        if not files:
            raise FileNotFoundError(f"No event files found for {cond} at {pattern}")

        # Group by containing directory: one SAC_<seed> folder == one run.
        runs = {}
        for f in files:
            runs.setdefault(os.path.dirname(f), []).append(f)

        data[cond] = [
            pd.concat([read_scalars(f) for f in sorted(runs[d])],
                      ignore_index=True)
            for d in sorted(runs)
        ]
        print(f"  {cond:17s} {len(data[cond])} runs "
              f"({', '.join(os.path.basename(d) for d in sorted(runs))})")
        # A run that stops early contributes nothing past its own last step;
        # the curves and asymptotic metrics silently rest on fewer seeds there.
        for d, df in zip(sorted(runs), data[cond]):
            last = df.step.max()
            if last < BUDGET:
                print(f"    ! {os.path.basename(d)} ends at {last:,} steps "
                      f"({100*last/BUDGET:.0f}% of the {BUDGET:,} budget)")
    return data


def series(df, tag):
    """Extract a single scalar tag as (steps, values), sorted by step."""
    s = df[df.tag == tag].sort_values("step")
    return s.step.values.astype(float), s.value.values.astype(float)


def ema(x, alpha=None):
    """Exponential moving average, used only for curve display.

    alpha is read from SMOOTH_ALPHA at call time (not bound as a default), so
    changing the constant actually takes effect.
    """
    if alpha is None:
        alpha = SMOOTH_ALPHA
    if len(x) == 0:
        return x
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


def millions_axis(ax, label="Environment timestep"):
    """Label the x axis in units of 1e6 steps (200000 -> 0.2).

    Decimal places follow the tick spacing so short axes (e.g. the 150k critic
    panel) do not collapse every tick to 0.0, and so 0 / 1 are not printed
    bare next to 0.2 / 0.4.
    """
    def fmt(v, _):
        ticks = ax.get_xticks()
        step = np.min(np.diff(ticks)) / 1e6 if len(ticks) > 1 else 0.1
        dp = max(1, int(np.ceil(-np.log10(step)))) if step > 0 else 1
        return f"{v/1e6:.{dp}f}"
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(fmt))
    if label:
        ax.set_xlabel(f"{label} " + r"($\times 10^{6}$)")


def resample(steps, values, grid):
    """Interpolate one run onto the common step grid (NaN outside its range)."""
    if len(steps) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    out = np.interp(grid, steps, values, left=np.nan, right=np.nan)
    out[grid < steps[0]] = np.nan
    out[grid > steps[-1]] = np.nan
    return out


def aggregate(runs, tag, grid, smooth=True):
    """IQM curve plus min/max envelope and the individual resampled runs."""
    mat = []
    for df in runs:
        st, va = series(df, tag)
        if smooth:
            va = ema(va)
        mat.append(resample(st, va, grid))
    mat = np.vstack(mat)
    # Curves run the full length of the longest run of a condition, but the
    # number of runs supporting each point is returned alongside so it can be
    # drawn. Runs end at different steps (Section 5.2 of the paper), so without
    # this the tail of a curve summarises fewer and fewer runs without saying
    # so. n_runs is what fig_overview plots in its lower strip.
    n_runs = np.sum(~np.isnan(mat), axis=0)
    iqm = np.array([
        trim_mean(col[~np.isnan(col)], 0.25) if np.sum(~np.isnan(col)) >= 2 else np.nan
        for col in mat.T
    ])
    return iqm, np.nanmin(mat, axis=0), np.nanmax(mat, axis=0), mat, n_runs


def return_scale(data):
    """Mean |episodic return| per condition over the asymptotic window.

    Used to normalise critic loss so conditions with different return
    magnitudes are comparable. Measured from the logs rather than hard-coded,
    since the scale changes whenever the reward or observation changes.
    """
    scale = {}
    for cond, runs in data.items():
        vals = []
        for df in runs:
            st, rw = series(df, "rollout/ep_rew_mean")
            vals.append(np.abs(rw[(st >= BUDGET - ASYM_WINDOW) & (st < BUDGET)]))
        pooled = np.concatenate(vals)
        scale[cond] = float(np.mean(pooled)) if len(pooled) else 1.0
        print(f"  {cond:17s} mean |return| = {scale[cond]:.1f}")
    return scale


# --------------------------------------------------------------------------
# Shared plotting helper
# --------------------------------------------------------------------------

def count_strip(ax, data, tag, conds, grid, xmax=None):
    """Draw how many runs support each point of the curves above.

    Runs of a condition end at different steps, so the right-hand part of a
    curve can rest on fewer runs than the left. Plotting the count states that
    explicitly instead of leaving it to the caption.
    """
    for cond in conds:
        _, _, _, _, n_runs = aggregate(data[cond], tag, grid)
        n = np.where(n_runs > 0, n_runs, np.nan)
        ax.step(grid, n, color=COLORS[cond], ls=STYLES[cond], lw=1.3, where="post")
    ax.set_ylabel("runs")
    ax.set_ylim(0, 5.6)
    ax.set_yticks([0, 5])
    ax.set_xlim(0, BUDGET if xmax is None else xmax)
    millions_axis(ax)


def curve_panel(ax, data, tag, conds, grid, ylabel, band=True, faint=True,
                xmax=None):
    for cond in conds:
        iqm, lo, hi, mat, _ = aggregate(data[cond], tag, grid)
        c, ls = COLORS[cond], STYLES[cond]
        if faint:
            for row in mat:
                ax.plot(grid, row, color=c, lw=0.4, alpha=0.16, zorder=1)
        if band:
            ax.fill_between(grid, lo, hi, color=c, alpha=0.13, lw=0, zorder=2)
        ax.plot(grid, iqm, color=c, ls=ls, lw=1.9, label=LABELS[cond], zorder=3)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, BUDGET if xmax is None else xmax)
    millions_axis(ax)


def save(fig, out, name):
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {name}.pdf / .png")


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_overview(data, grid, out):
    """Fig. 2 - all four conditions on the headline metrics.

    Each metric is written as its own standalone figure so the panels can be
    placed independently in the paper. A panel may specify its own x-extent;
    the return panel is drawn to 1M steps to show the full training run.
    """
    conds = list(CONDITIONS)
    panels = [
        ("episode/laps_completed", "Laps per episode", "Task competence",
         "upper left", "fig2a_all_conditions_laps", None),
        ("rollout/ep_len_mean", "Mean episode length (steps)", "Episode survival",
         "upper left", "fig2b_all_conditions_eplen", None),
        ("rollout/ep_rew_mean", "Mean episodic return", "Episodic return",
         "upper left", "fig2c_all_conditions_reward", PLOT_XMAX),
    ]
    for tag, ylabel, title, legend_loc, name, xmax in panels:
        # Extend the resampling grid to match a wider axis, otherwise the
        # curves would stop at BUDGET and leave the panel blank on the right.
        g = grid if xmax is None else np.linspace(0, xmax, int(450 * xmax / BUDGET))
        # A short strip under each panel reports how many runs support each
        # point, since runs of a condition end at different steps.
        fig, (ax, axn) = plt.subplots(
            2, 1, figsize=(5.4, 4.1), sharex=True,
            gridspec_kw={"height_ratios": [4.6, 1.0], "hspace": 0.08})
        curve_panel(ax, data, tag, conds, g, ylabel, xmax=xmax)
        ax.set_title(title)
        # A single narrow panel leaves less room than the old two-panel layout,
        # so add headroom to keep the legend clear of the curves.
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, lo + (hi - lo) * 1.32)
        ax.legend(loc=legend_loc, frameon=False, fontsize=7)
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
        count_strip(axn, data, tag, conds, g, xmax=xmax)
        save(fig, out, name)


def fig_pair(data, grid, out, pair):
    """Fig. 3 / 4 / 5 - one ablation axis at a time, with early-learning inset."""
    ref, var, name, title = PAIRS[pair]
    conds = [ref, var]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    curve_panel(axes[0], data, "episode/laps_completed", conds, grid, "Laps per episode")
    curve_panel(axes[1], data, "rollout/ep_len_mean", conds, grid, "Mean episode length (steps)")
    axes[0].set_title("(a) Task competence")
    axes[1].set_title("(b) Episode survival")
    axes[0].legend(loc="upper left", frameon=False)

    # Inset over the first 50k steps, where early-learning differences live.
    ins = axes[1].inset_axes([0.46, 0.14, 0.50, 0.44])
    gsub = np.linspace(0, 50_000, 260)
    for cond in conds:
        iqm, lo, hi, _, _ = aggregate(data[cond], "rollout/ep_len_mean", gsub)
        ins.fill_between(gsub, lo, hi, color=COLORS[cond], alpha=0.15, lw=0)
        ins.plot(gsub, iqm, color=COLORS[cond], ls=STYLES[cond], lw=1.5)
    ins.set_title("first 50k steps", fontsize=7, pad=2)
    ins.tick_params(labelsize=6)
    ins.grid(alpha=0.2)
    millions_axis(ins, label=None)

    fig.suptitle(title, fontsize=10, y=1.02)
    save(fig, out, name)


def fig_laptimes(data, out):
    """Fig. 6 - lap-time distributions: consistency, not peak speed."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    # The CNN policy completes almost no laps, and none inside the asymptotic
    # window, so it has no lap-time distribution to draw (it is skipped below).
    groups = [(["AE32_nospeed", "AE32_speed"], "(a) AE-32: effect of speed input"),
              (["AE32_speed", "AE64_speed"], "(b) With speed: latent size")]
    for ax, (conds, title) in zip(axes, groups):
        for cond in conds:
            pool = []
            for df in data[cond]:
                st, lt = series(df, "time/lap_time")
                keep = (lt > LAP_MIN_VALID) & (st >= BUDGET - ASYM_WINDOW) & (st < BUDGET)
                pool.append(lt[keep])
            pool = np.concatenate(pool)
            if len(pool) == 0:
                continue
            xs = np.sort(pool)
            ys = np.arange(1, len(xs) + 1) / len(xs)
            ax.plot(xs, ys, color=COLORS[cond], ls=STYLES[cond], lw=1.9, label=LABELS[cond])
        ax.set_xlabel("Lap time (s)")
        ax.set_ylabel("Cumulative fraction of laps")
        ax.set_xlim(25, 50)
        ax.set_title(title)
        ax.legend(loc="lower right", frameon=False)
    save(fig, out, "fig6_laptime_ecdf")


def fig_entropy(data, grid, out):
    """Fig. 7 - alpha adapts to the state representation."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.3))
    conds = list(CONDITIONS)
    curve_panel(axes[0], data, "train/ent_coef", conds, grid,
                r"Entropy coefficient $\alpha$", faint=False)
    curve_panel(axes[1], data, "train/std", conds, grid,
                "Policy standard deviation", faint=False)
    axes[0].set_yscale("log")
    axes[0].set_title(r"(a) Entropy temperature")
    axes[1].set_title("(b) Policy stochasticity")
    axes[0].legend(loc="upper right", frameon=False, ncol=2, fontsize=7)
    save(fig, out, "fig7_entropy")


def fig_critic(data, scale, out):
    """Fig. 8 - early critic loss, normalised by each condition's return scale."""
    grid = np.linspace(0, 150_000, 400)
    fig, ax = plt.subplots(figsize=(5.4, 3.4))
    peaks = []
    for cond in CONDITIONS:
        iqm, lo, hi, _, _ = aggregate(data[cond], "train/critic_loss", grid)
        s = scale[cond]
        ax.fill_between(grid, lo / s, hi / s, color=COLORS[cond], alpha=0.12, lw=0)
        ax.plot(grid, iqm / s, color=COLORS[cond], ls=STYLES[cond], lw=1.9,
                label=LABELS[cond])
        finite = iqm[np.isfinite(iqm) & (iqm > 0)]
        if len(finite):
            peaks.append(np.nanmax(finite) / s)
    # A condition that never learns has a near-zero return scale, which inflates
    # its normalised loss by orders of magnitude. Switch to log so the
    # better-performing conditions stay readable instead of collapsing onto 0.
    if len(peaks) > 1 and max(peaks) / min(peaks) > 20:
        ax.set_yscale("log")
    ax.set_ylabel("Critic loss / condition mean |return|")
    ax.set_title("Value-function error during early training")
    ax.set_xlim(0, 150_000)
    millions_axis(ax)
    ax.legend(frameon=False, fontsize=7)
    save(fig, out, "fig8_critic_loss")


def fig_summary(metrics, out):
    """Fig. 9 - effect of each ablation by horizon."""
    def iqm_of(cond, col):
        v = metrics.loc[metrics.cond == cond, col].dropna()
        return trim_mean(v, 0.25) if len(v) else np.nan

    cols = ["js_eplen", "js_laps", f"ttt{THRESHOLD}", "asym_eplen", "asym_laps"]
    names = ["Early\n(ep. length)", "Early\n(laps)", "Convergence\nspeed",
             "Asymptotic\n(ep. length)", "Asymptotic\n(laps)"]

    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    pairs = list(PAIRS.values())
    width, xs = 0.8 / len(pairs), np.arange(len(cols))
    for i, (ref, var, _, _) in enumerate(pairs):
        off = (i - (len(pairs) - 1) / 2) * width
        vals = []
        for c in cols:
            num, den = iqm_of(var, c), iqm_of(ref, c)
            r = num / den if den not in (0, np.nan) and np.isfinite(den) and den != 0 else np.nan
            if c.startswith("ttt"):      # lower is better -> invert to a speedup
                r = 1.0 / r if r and np.isfinite(r) and r != 0 else np.nan
            vals.append(100 * (r - 1) if np.isfinite(r) else np.nan)
        ax.bar(xs + off, vals, width,
               color=COLORS[var], alpha=0.85,
               label=f"{LABELS[var]}  vs  {LABELS[ref]}",
               edgecolor="white", linewidth=0.6)
        for x, v in zip(xs + off, vals):
            if np.isfinite(v):
                ax.text(x, v + (14 if v >= 0 else -22), f"{v:+.0f}%",
                        ha="center", fontsize=6.5)
            else:
                # e.g. time-to-threshold when a condition never reaches it:
                # undefined, not zero - say so rather than leaving a gap.
                ax.text(x, 0, "n/a", ha="center", va="bottom", fontsize=6.5,
                        color="0.45", rotation=90)
    ax.axhline(0, color="0.25", lw=0.9)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("Change relative to reference condition (%)")
    ax.set_yscale("symlog", linthresh=50)
    ax.set_title("Ablation effect by horizon")
    ax.legend(frameon=False, fontsize=7)
    save(fig, out, "fig9_ablation_summary")


# --------------------------------------------------------------------------
# Metrics table
# --------------------------------------------------------------------------

def compute_metrics(data):
    rows = []
    for cond, runs in data.items():
        for i, df in enumerate(runs):
            st, el = series(df, "rollout/ep_len_mean")
            sl, lp = series(df, "episode/laps_completed")
            slt, lt = series(df, "time/lap_time")
            sr, rw = series(df, "rollout/ep_rew_mean")
            se, ec = series(df, "train/ent_coef")
            _, sd = series(df, "train/std")

            keep = lt > LAP_MIN_VALID
            slt, lt = slt[keep], lt[keep]
            win = lambda s, v, a, b: v[(s >= a) & (s < b)]
            fin = lambda s, v: win(s, v, BUDGET - ASYM_WINDOW, BUDGET)
            hit = np.where((el >= THRESHOLD) & (st < BUDGET))[0]
            fl = fin(slt, lt)

            rows.append(dict(
                cond=cond, run=i,
                js_eplen=np.mean(win(st, el, 0, JUMP_WINDOW)),
                js_laps=np.mean(win(sl, lp, 0, JUMP_WINDOW)),
                **{f"ttt{THRESHOLD}": st[hit[0]] if len(hit) else np.nan},
                asym_eplen=np.mean(fin(st, el)),
                asym_laps=np.mean(fin(sl, lp)),
                asym_rew=np.mean(fin(sr, rw)),
                rew_per_step=np.mean(fin(sr, rw)) / np.mean(fin(st, el)),
                best_lap=lt.min() if len(lt) else np.nan,
                med_lap=np.median(fl) if len(fl) else np.nan,
                worst_lap=lt.max() if len(lt) else np.nan,
                total_laps=lp[sl < BUDGET].sum(), n_episodes=int((sl < BUDGET).sum()),
                ent_first=ec[0], ent_final=np.mean(fin(se, ec)), std_first=sd[0],
            ))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="./logs",
                    help="directory containing the four condition folders")
    ap.add_argument("--out", default="./figures", help="output directory")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("Loading event files...")
    data = load_runs(args.root)

    print("Measuring return scale...")
    scale = return_scale(data)

    print("Computing metrics...")
    metrics = compute_metrics(data)
    metrics.to_csv(os.path.join(args.out, "per_run_metrics.csv"), index=False)
    summary = metrics.drop(columns=["run"]).groupby("cond").agg(
        lambda x: trim_mean(x.dropna(), 0.25))
    summary.to_csv(os.path.join(args.out, "condition_iqm.csv"))
    print(summary.round(3).to_string())

    print("Generating figures...")
    grid = np.linspace(0, BUDGET, 450)
    fig_overview(data, grid, args.out)
    for pair in PAIRS:
        fig_pair(data, grid, args.out, pair)
    fig_laptimes(data, args.out)
    fig_entropy(data, grid, args.out)
    fig_critic(data, scale, args.out)
    fig_summary(metrics, args.out)
    print(f"\nDone. Figures and CSVs in {args.out}/")


if __name__ == "__main__":
    main()
