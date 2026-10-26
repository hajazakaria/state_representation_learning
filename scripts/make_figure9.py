#!/usr/bin/env python3
"""Regenerate Figure 9 of the paper (lap-time distributions) from the released logs.

Run traceable_metrics.py first; this script reads the lap events it writes.

    python scripts/traceable_metrics.py --root logs/logs --out stats
    python scripts/make_figure9.py --laps stats/laps_traceable.csv --out figure_9_lap_times.jpg

Lap events are those logged before 900k steps, excluding events shorter than
15 s, pooled over the five runs of each configuration. Configuration 1 is
omitted because it recorded no sustained lap completion (Table 4, note a).

Part of the code and data release for Haja et al., Array, 2026. Licensed MIT.
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ORDER = [("Config 2", "Config 2\nCAE 32-dim"),
         ("Config 3", "Config 3\nCAE 32-dim + Speed"),
         ("Config 4", "Config 4\nCAE 64-dim + Speed")]
COLORS = ["#aec7e8", "#ffbb78", "#98df8a"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--laps", default="stats/laps_traceable.csv")
    ap.add_argument("--out", default="figure_9_lap_times.jpg")
    args = ap.parse_args()

    laps = pd.read_csv(args.laps)
    plt.rcParams.update({"font.family": "serif",
                         "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                         "font.size": 11})
    data = [laps[laps.config == c].lap_time.values for c, _ in ORDER]
    fig, ax = plt.subplots(figsize=(9.4, 7.0))
    bp = ax.boxplot(data, patch_artist=True, widths=0.3,
                    medianprops=dict(color="black", lw=1.4),
                    flierprops=dict(marker="o", markerfacecolor="none", markersize=8,
                                    markeredgecolor="black"))
    for patch, col in zip(bp["boxes"], COLORS):
        patch.set_facecolor(col)
        patch.set_edgecolor("black")
    ax.set_xticks(range(1, len(ORDER) + 1))
    ax.set_xticklabels([label for _, label in ORDER])
    ax.set_ylabel("Lap Time (seconds)", fontsize=14)
    ax.set_title("Lap Time Distribution across Representations", fontsize=15)
    ax.yaxis.grid(True, ls="--", alpha=0.7)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(args.out, dpi=500)
    for (c, _), d in zip(ORDER, data):
        print("%s  laps=%d  fastest=%.2f s  median=%.2f s" % (c, len(d), d.min(), pd.Series(d).median()))


if __name__ == "__main__":
    main()
