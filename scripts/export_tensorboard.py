"""Export TensorBoard event files to a tidy CSV of learning curves.

Why this exists
---------------
Event files are the raw record and belong in the repository, but reading them
requires TensorFlow or the TensorBoard package and knowledge of the tag names.
A single long-format CSV --

    config, run, step, metric, value

-- lets any reader recompute every entry of Table 3, redraw Figure 8, and run
their own statistical tests across the five runs of a configuration without
installing anything beyond pandas. Reviewer 2 asked for the per-run logs
specifically so that the comparisons in the paper could be checked; this is the
artefact that makes that possible.

Usage
-----
    python scripts/export_tensorboard.py --logdir logs --out logs/eval_curves.csv

    # see which scalar tags your runs recorded, then filter
    python scripts/export_tensorboard.py --logdir logs --list-tags
    python scripts/export_tensorboard.py --logdir logs --tags eval/mean_reward

Directory layout expected
-------------------------
    logs/config3/run2/events.out.tfevents...

Config and run are inferred from the path: any component matching ``config<N>``
or ``run<N>`` / ``seed<N>`` is picked up. Anything unrecognised is reported so
you can rename rather than silently mislabel a curve.

Part of the code and data release for Haja et al., Array (under review), 2026.
Licensed MIT.
"""

import argparse
import os
import re
import sys
from collections import OrderedDict

CONFIG_RE = re.compile(r"^config[_-]?(\d+)$", re.I)
RUN_RE = re.compile(r"^(?:run|seed)[_-]?(\d+)$", re.I)


def find_run_dirs(logdir):
    """Yield every directory containing at least one TensorBoard event file."""
    for root, _dirs, files in os.walk(logdir):
        if any(f.startswith("events.out.tfevents") for f in files):
            yield root


def parse_ids(path, logdir):
    """Infer (config, run) from a run directory path."""
    rel = os.path.relpath(path, logdir)
    parts = [p for p in rel.replace("\\", "/").split("/") if p and p != "."]
    config = run = None
    for p in parts:
        m = CONFIG_RE.match(p)
        if m:
            config = int(m.group(1))
            continue
        m = RUN_RE.match(p)
        if m:
            run = int(m.group(1))
    return config, run, rel


def read_scalars(run_dir, wanted_tags=None):
    """Return {tag: [(step, value), ...]} for one run directory."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    acc = EventAccumulator(run_dir, size_guidance={"scalars": 0})  # 0 = load all
    acc.Reload()
    tags = acc.Tags().get("scalars", [])
    if wanted_tags:
        tags = [t for t in tags if t in wanted_tags]
    return OrderedDict((t, [(e.step, e.value) for e in acc.Scalars(t)]) for t in tags)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logdir", default="logs", help="root of the log tree")
    ap.add_argument("--out", default="logs/eval_curves.csv", help="output CSV")
    ap.add_argument("--tags", nargs="*", default=None,
                    help="scalar tags to export (default: all)")
    ap.add_argument("--list-tags", action="store_true",
                    help="print the tags present in each run and exit")
    args = ap.parse_args()

    run_dirs = sorted(find_run_dirs(args.logdir))
    if not run_dirs:
        print("No event files found under %r." % args.logdir, file=sys.stderr)
        print("Extract the archive into logs/ first -- see logs/README.md.",
              file=sys.stderr)
        return 1

    print("found %d run directories under %s" % (len(run_dirs), args.logdir))

    if args.list_tags:
        for d in run_dirs:
            tags = read_scalars(d).keys()
            print("\n%s" % os.path.relpath(d, args.logdir))
            for t in tags:
                print("   ", t)
        return 0

    rows = []
    unlabelled = []
    for d in run_dirs:
        config, run, rel = parse_ids(d, args.logdir)
        if config is None or run is None:
            unlabelled.append(rel)
        scalars = read_scalars(d, args.tags)
        n = 0
        for tag, points in scalars.items():
            for step, value in points:
                rows.append((config, run, rel, step, tag, value))
                n += 1
        print("  %-40s config=%-4s run=%-4s points=%d"
              % (rel, config, run, n))

    if unlabelled:
        print("\nCould not infer config/run for %d directory(ies):"
              % len(unlabelled), file=sys.stderr)
        for r in unlabelled:
            print("   ", r, file=sys.stderr)
        print("Rename them to the config<N>/run<M> layout so the curves are "
              "labelled correctly rather than left blank.", file=sys.stderr)

    try:
        import pandas as pd

        df = pd.DataFrame(
            rows, columns=["config", "run", "source", "step", "metric", "value"]
        )
        df.sort_values(["config", "run", "metric", "step"], inplace=True)
        df.to_csv(args.out, index=False)
    except ImportError:
        import csv

        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["config", "run", "source", "step", "metric", "value"])
            w.writerows(sorted(rows, key=lambda r: (r[0] or 0, r[1] or 0, r[4], r[3])))

    print("\nwrote %d rows to %s" % (len(rows), args.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
