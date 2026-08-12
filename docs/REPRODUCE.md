# Reproducing the paper

Every number in the paper traces to one of the four steps below. Wall-clock
figures are for the machine in Section 5.1 — an Intel Core i5-12500 with no
discrete GPU, CPU-only throughout.

---

## 0. Environment

```bash
conda env create -f environment.yml
conda activate donkeyenv2
pip install git+https://github.com/araffin/gym-donkeycar-1.git
```

The experiments used the **`araffin/gym-donkeycar-1` fork**, not the upstream
`tawnkramer/gym-donkeycar`; the two differ in environment configuration
defaults. The fork reports package version 21.02.20, inherited from the release
it branched from.

Download the DonkeySim binary, point the Gym environment config at it, and select
the **MiniMonaco** track.

Do not upgrade `gym` or `stable-baselines3`. SB3 1.5.0 predates the Gym 0.26
step-API change, and upgrading either produces a tuple-unpacking error on the
first `env.step()`.

---

## 1. CAE pre-training → Figures 6 and 7, and the SSIM values in Section 6.1

Both encoders in `models/` were produced this way -- `ae-32_minimonaco_600_epochs_best.pkl`
and its 64-dim counterpart. Re-running is only necessary if you want to verify
training; to reproduce the RL results, use the shipped weights.

| Setting | Value |
|---|---|
| Epochs | up to 600 |
| Optimiser | Adam, lr 1e-3, β = (0.9, 0.999) |
| Batch size | 32 |
| Split | 80 / 20, **frame-level** |
| Early stopping | patience 30 on validation MSE, best-epoch weights restored |
| Loss | MSE; SSIM (scikit-image) tracked as a reconstruction metric |

Expected: validation SSIM ≈ 0.625 for 32-dim, ≈ 0.675 for 64-dim. Runtime
≈ 4 h 17 min and ≈ 33 h 47 min respectively.

The 64-dim model reconstructs better on both metrics. The point of the paper is
that this does not carry over to control performance.

---

## 2. SAC training → Figure 8 and Table 3

Four configurations, multiple independent runs each, 1,000,000 environment steps
per run. (The run count in `logs/` must match the figure stated in Section 5.2 of
the paper -- see the note in the top-level README.)

| Config | Policy | Observation fed to the policy |
|---|---|---|
| 1 | `CnnPolicy` (NatureCNN, trained from scratch online) | raw 120×160×3 frame |
| 2 | `MlpPolicy` | `z_t` from `ae-32_minimonaco_600_epochs_best.pkl` |
| 3 | `MlpPolicy` | `[z_t, v_t]` from `ae-32_minimonaco_600_epochs_best.pkl` |
| 4 | `MlpPolicy` | `[z_t, v_t]` from `ae-64_minimonaco_600_epochs_best.pkl` |

Hyperparameters are the RL Zoo SAC defaults, unchanged across all four — see the
table in the top-level README. The encoder is frozen for Configs 2–4; only the
actor, critics and entropy temperature are updated.

Wrap the environment so that each `step` returns the state vector rather than
the frame: encode the frame with `load_ae(...).encode_from_raw_image(frame)`
from `src/autoencoder/autoencoder_32.py` (or `autoencoder_64.py`), and for
Configs 3 and 4 append the simulator speed divided by `v_max = 10.0` and clipped
to `[0, 1]`. Apply the
reward from Equation 5, and terminate on collision or `|CTE| > max_cte = 8.0`.
Steering is `[-1, 1]`; throttle is rescaled to `[0, 1]` before it reaches the
simulator.

Evaluation: every 10,000 steps, 5 deterministic episodes from the same fixed
start pose. Evaluation transitions must not enter the replay buffer.

Cost ≈ 180 s per 10k steps for Configs 2–4, ≈ 430 s for Config 1 — so a single
Config 1 run is roughly 12 hours and a full sweep across all four
configurations and every run is on the order of a week of CPU time. **This excludes the ≈ 38 hours of CAE pre-training that
Configs 2–4 additionally require and Config 1 does not.** The paper reports both
figures rather than the RL time alone.

---

## 3. Figures and tables

Regenerate the curves from the shipped logs, with no retraining:

```bash
python scripts/make_figures.py --root logs --out ./figures
```

`make_figures.py` parses the event files directly, so no TensorFlow is needed.
Curves are exponentially smoothed with `SMOOTH_ALPHA = 0.20` -- the value quoted
in the Figure 8 caption -- and resampled onto a common step grid before
aggregation. `scripts/export_tensorboard.py` will additionally emit a tidy
`config, run, step, metric, value` CSV if you want the curves as plain text.

Figure 8 shows, per configuration: the **interquartile mean** across the five
runs as a solid curve, the **min–max envelope** of the individual runs as a
shaded band, and the individual run trajectories faintly behind the aggregate.
Runs are exponentially smoothed and resampled onto a common step grid before
aggregation, so the band reflects between-run spread rather than differences in
logging cadence.

Metric definitions, which matter because two of them are easy to confuse:

| Metric | Definition |
|---|---|
| Mean Reward | mean ± SD of the final evaluation return across the 5 runs |
| Peak Reward | single highest evaluation return at any evaluation point in any run — an extremum, so **higher than the peak of the aggregated curve** in Figure 8 |
| Best Lap | fastest single lap across all evaluation episodes and all runs; N/A if no lap was completed |
| Median Lap | median over all laps completed during post-convergence evaluation episodes, pooled across runs |
| Conv. Step | first step at which a configuration's evaluation return reaches 50% of *its own* peak — a curve-shape measure, not an absolute level |

Reference values from Table 3: Config 3 peaks at 4,084 with a 26.2 s best lap
and a 28.5 s median, reaching 50%-of-peak at ≈ 136k steps. Config 4 peaks at
3,102, reaching 50%-of-peak at ≈ 153k. Config 1 never completed a lap.

---

## 4. What cannot be reproduced from this repository

- **Cross-track results.** There are none. Everything is MiniMonaco.
- **Statistical tests.** None were run; per-run results are reported
  descriptively. The logs are provided so anyone can compute their own.
- **A competitive end-to-end baseline.** Config 1 is SB3 defaults with no image
  augmentation. It is a reference point, not a tuned pixel-based agent.

---

## BibTeX

```bibtex
@article{haja2026impact,
  title   = {The Impact of State Representation on Entropic Control for Autonomous Racing},
  author  = {Haja, Zakaria and Haja, Fatima-Zahra and Kelmoua, Leila and
             Bouchentouf, Toumi and Berrich, Jamal},
  journal = {Array},
  year    = {2026},
  note    = {Under review}
}

@inproceedings{haja2026entropic,
  title     = {Entropic Control in Latent Spaces: Autonomous Racing in the
               DonkeyCar MiniMonaco Environment using Soft Actor-Critic},
  author    = {Haja, Zakaria and Haja, Fatima-Zahra and Kelmoua, Leila and
               Bouchentouf, Toumi and Berrich, Jamal},
  booktitle = {2026 International Conference on Artificial Intelligence for
               Sustainable Engineering and Innovation (AISEI)},
  pages     = {655--659},
  year      = {2026},
  doi       = {10.1109/AISEI68628.2026.11572730}
}
```
