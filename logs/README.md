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
