# Per-run training logs

TensorBoard event files for the 5 independent runs of each of the 4
configurations. These are the data behind Figure 8, Figure 9 and 
every entry in Table 3.

## Layout

```
logs/
├── config1/run1/ ... run5/     events.out.tfevents.*
├── config2/run1/ ... run5/
├── config3/run1/ ... run5/
└──config4/run1/ ... run5/



## Metric definitions

Two pairs are easy to confuse, so they are defined here as well as in the paper:

| Metric | Definition |
|---|---|
| Mean Reward | mean ± SD of the final evaluation return across the 5 runs |
| Peak Reward | highest evaluation return at any evaluation point in any run. An extremum, therefore **higher than the peak of the aggregated curve** in Figure 8, which plots a between-run summary statistic |
| Best Lap | fastest single lap across all evaluation episodes and all runs |
| Median Lap | median over all laps completed in post-convergence evaluation episodes, pooled across runs |
| Conv. Step | first step at which a configuration's evaluation return reaches 50% of *its own* peak. A curve-shape measure, not an absolute level |

## Lap times

`lap_times/` should record the configuration and run for every lap, so that both
Best Lap and Median Lap can be recomputed from source rather than taken on trust.

---

## Known properties of these logs

Two things a reader should know before using these files, both disclosed in
Section 7.3 of the paper.

**Three runs per configuration share an episode-boundary structure.** Within each
configuration, three of the five runs record an identical sequence of logged
steps across the whole of training, and coincide exactly on every metric over
part of the final phase (150 logged points for Config 1, 10 for Config 2, 5 for
Config 3). The runs are otherwise clearly distinct -- they differ in more than
95% of their logged returns and carry independent timestamps recorded days
apart -- so these are not duplicated files. We could not establish
retrospectively what produced the shared structure. Consequently the five runs
of a configuration should not be treated as fully independent replicates, and
inferential tests that assume independence are not reported in the paper.

**Run lengths differ from the configured budget.** Training was configured for
1,000,000 steps per run; no run reached exactly that figure, and one Config 1 run
stopped at 207,106. Per-run lengths are listed in Section 5.2 of the paper. One consequence
is visible in `per_run_metrics.csv`: the asymptotic window is the absolute range
[800,000, 900,000), and the Config 1 run that stopped early contributes no data
there, so that configuration's summary is reported as NaN rather than computed
over a different part of training.

## What is not here

The outputs of the periodic policy evaluations -- every 10,000 steps, five
episodes from a fixed start pose -- are the source of every number in Table 3 of
the paper, and they are **not** included in this repository. They were not
retained as part of the archived material for this study.

The practical consequence: Figure 8 and the reconstruction results can be
regenerated from what is here, and `run_statistics.py` recomputes the per-run
training metrics, but Table 3 cannot be recomputed from these files. The two
records are distinct and are not interchangeable. The training rollout return
logged here is measured under the stochastic exploration policy over a moving
window of recent episodes; the evaluation return in Table 3 is measured under
the deterministic policy from a fixed start pose. Comparing a number from one
against a number from the other is not meaningful.
