# Training logs and derived results

`logs.zip` holds the TensorBoard event files of all twenty reinforcement
learning runs, five per configuration. **Every reinforcement learning result in
the paper -- Figure 8, Figure 9, Table 4 and Table 5 -- is computed from these
files**, by the scripts in `scripts/`. They are the record of the experiments
reported in the paper.

## Layout

```
logs/
├── logs.zip                    raw TensorBoard event files (unzip first)
│   └── logs/
│       ├── cnnPolicy-minimonacoTrack/                  Config 1
│       ├── car_srl_ae_32_no_speed_minimonacoTrack/     Config 2
│       ├── car_srl_ae_32_with_speed_minimonacoTrack/   Config 3
│       └── car_srl_ae_64_with_speed_minimonacoTrack/   Config 4
│           └── donkey-minimonaco-track-v0/SAC_<n>/events.out.tfevents.*
├── per_run_traceable.csv       every run-level metric behind Table 4
├── per_run_laps.csv            per-run lap statistics behind Table 4
├── laps_traceable.csv          every lap event used for Table 4 and Figure 9
├── table4_summary.csv          the across-run summary printed in Table 4
├── table5_sensitivity.csv      the collapsed-run summary printed in Table 5
└── per_run_metrics.csv         output of run_statistics.py
```

One `SAC_*` directory is one run.

## Reproducing Tables 4 and 5 and Figure 9

```bash
unzip logs/logs.zip -d logs/
python scripts/traceable_metrics.py --root logs/logs --out stats
python scripts/make_figure9.py --laps stats/laps_traceable.csv
```

The CSV files committed in this folder are exactly what `traceable_metrics.py`
writes; they are provided so the numbers can be checked without running anything.

## How the metrics are computed

The aggregation is the same for every metric. **Within each run**, the metric is
reduced to one run-level value. **Across runs**, those values are summarised as
mean ± standard deviation, and the number of contributing runs is given whenever
it is fewer than five.

| Metric | Run-level value |
|---|---|
| Return 0.1–0.2M | mean of `rollout/ep_rew_mean` over steps [100,000, 200,000) |
| Return 0.8–0.9M | mean of `rollout/ep_rew_mean` over steps [800,000, 900,000) |
| Peak return | maximum unsmoothed `rollout/ep_rew_mean` before step 900,000 |
| Laps/ep. 0.8–0.9M | mean of `episode/laps_completed` over steps [800,000, 900,000) |
| Steps to 1,000 | first logged step at which `rollout/ep_rew_mean` ≥ 1,000 |
| Best / median lap, lap IQR | from `time/lap_time` events before step 900,000, excluding events under 15 s |

The windows are absolute step ranges, identical for every run. A run contributes
to a window only if it reaches the end of it. The early window is reached by all
twenty runs, so it is unaffected by the unequal training budgets described below.

`rollout/ep_rew_mean` is the training rollout return: the running mean episodic
return logged throughout training under the stochastic exploration policy.

## Known properties of these logs

Both are disclosed in Section 7.3 of the paper.

**Three runs per configuration share a logging structure.** Within each
configuration, three of the five runs share an identical sequence of logging
steps -- the environment steps at which metrics, including lap events, were
written -- over the whole of training:

| Configuration | Runs sharing the structure |
|---|---|
| Config 1 | SAC_1, SAC_4, SAC_5 |
| Config 2 | SAC_3, SAC_4, SAC_5 |
| Config 3 | SAC_2, SAC_4, SAC_5 |
| Config 4 | SAC_1, SAC_4, SAC_5 |

Over the final stretch of training the three runs record identical values
across every logged metric (the last 150 logged points for Config 1, 10 for
Config 2, 5 for Config 3 and 1 for Config 4). Before that stretch, their logged
returns, episode lengths and training losses coincide in fewer than 2% of points,
and their returns at no more than two isolated points per pair of runs. The only
other coincidences are among Config 1's few lap events, most of which are timing
artefacts shorter than 15 s. Because the logged episode lengths differ, the shared
sequence is one of logging steps, not of episode boundaries. The runs carry
independent timestamps recorded days apart and are not duplicated files. We
could not establish retrospectively what produced the shared structure.

Consequently the five runs of a configuration are not five independent
replicates. Table 5 of the paper, written to `table5_sensitivity.csv`, repeats
the main comparison with the three runs averaged into one; the ordering of
configurations is unchanged. For Configs 2–4 the identical final stretches lie
beyond step 900,000, outside every window used in Table 4.

**Run lengths differ from the configured budget.** Training was configured for
1,000,000 steps per run. No run reached exactly that figure, and one Config 1
run (SAC_3) stopped at 207,106. Per-run lengths are in Table 3 of the paper. This
is why every comparison uses fixed step windows rather than the final step.

## What is not here

During training the policy was also evaluated every 10,000 steps over five
deterministic episodes. Those evaluation outputs were not retained, and no
result in the paper depends on them. The SAC training scripts and the trained
policies are not part of this release either; see `docs/REPRODUCE.md` for the
upstream projects the training was run with.

Because random seeds were not fixed, re-running training will not reproduce
these runs bit for bit. The event files here are the record.
