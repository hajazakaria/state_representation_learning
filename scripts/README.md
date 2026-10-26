# Scripts

| File | Purpose |
|---|---|
| `traceable_metrics.py` | computes every entry of Tables 4 and 5 of the paper from the event files; see `logs/README.md` for the definitions |
| `make_figure9.py` | regenerates Figure 9 from the lap events written by `traceable_metrics.py` |
| `make_figures.py` | regenerates the learning curves of Figure 8 directly from the event files, plus supplementary diagnostic plots |
| `run_statistics.py` | per-run asymptotic returns over [800k, 900k); its pairwise tests assume independent runs and are not reported in the paper |
| `export_tensorboard.py` | converts the event files into one tidy long-format CSV (`config, run, step, metric, value`) |
| `inspect_checkpoint.py` | prints the contents of a `models/*.pkl` checkpoint -- latent width, layer names, and whether it holds the encoder alone or the full autoencoder |

## Regenerating the figures

```bash
pip install matplotlib pandas numpy scipy
python make_figures.py --root ../logs --out ./figures
```

Event files are parsed directly, so neither TensorFlow nor tbparse is needed.
One `SAC_*` directory is treated as one run; if such a directory holds several
event files, for instance after a resumed run, they are concatenated.

Key constants, at the top of `make_figures.py`:

| Constant | Value | Meaning |
|---|---|---|
| `SMOOTH_ALPHA` | 0.20 | exponential moving average weight; equivalent to a TensorBoard smoothing slider of 0.8 |
| `PLOT_XMAX` | 1,000,000 | x-extent of the return curve, matching the training budget in Table 2 |
| `ASYM_WINDOW` | 100,000 | final-performance window (last 100k steps) |
| `JUMP_WINDOW` | 10,000 | early-learning window (first 10k steps) |
| `LAP_MIN_VALID` | 15.0 s | laps below this are treated as detection artefacts |

`SMOOTH_ALPHA` is the smoothing window quoted in the Figure 8 caption of the
paper. If you change it here, change it there too.

Aggregation follows Agarwal et al. (2021): the interquartile mean across runs,
with a min-max envelope as a shaded band and individual runs drawn faintly
behind. Curves are resampled onto a common step grid before aggregation, so runs
with different logging cadences combine correctly.

## Not included

`train_cae.py`, `train_sac.py` and `evaluate_policy.py`. Until they are added,
the reproduction path for RL training routes through araffin/aae-train-donkeycar
as documented in `docs/REPRODUCE.md`.
