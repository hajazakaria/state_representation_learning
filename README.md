# State Representation Learning for Autonomous Racing with Soft Actor-Critic

Code, pre-trained convolutional autoencoders, and the image corpus supporting the paper:

> **The Impact of State Representation on Entropic Control for Autonomous Racing**
> Z. Haja, F.-Z. Haja, L. Kelmoua, T. Bouchentouf, J. Berrich.
> Submitted to *Array* (Elsevier), 2026. Manuscript ARRAY-D-26-02717.

This work extends our earlier study:

> **Entropic Control in Latent Spaces: Autonomous Racing in the DonkeyCar MiniMonaco Environment using Soft Actor-Critic**
> 2026 International Conference on Artificial Intelligence for Sustainable Engineering and Innovation (AISEI), pp. 655–659.
> DOI: [10.1109/AISEI68628.2026.11572730](https://doi.org/10.1109/AISEI68628.2026.11572730)

---

## What this study compares

A Soft Actor-Critic agent is trained on the DonkeyCar **MiniMonaco** circuit under four state representations. Everything else — algorithm, reward, hyperparameters, training budget, evaluation protocol — is held fixed.

| Config | State $s_t$ | Dim | Policy | Encoder |
|---|---|---|---|---|
| 1 | Raw RGB frame | 120×160×3 | `CnnPolicy` (NatureCNN) | learned online, jointly with the policy |
| 2 | $z_t$ | 32 | `MlpPolicy` | frozen CAE-32 |
| 3 | $[z_t, v_t]$ | 33 | `MlpPolicy` | frozen CAE-32 (same weights as Config 2) |
| 4 | $[z_t, v_t]$ | 65 | `MlpPolicy` | frozen CAE-64 |

$z_t$ is the latent code of the current frame; $v_t$ is the scalar forward speed, scaled to $[0,1]$ by division by $v_{\max}=10$.

**Headline result.** Config 3 — the *smaller* latent space with speed — performs best, despite the 64-dim autoencoder reconstructing images more faithfully (SSIM 0.675 vs 0.625). Reconstruction fidelity and control utility are not the same objective.

**Scope.** All experiments are on a single circuit. Cross-track generalisation is untested and the results should be read as track-specific. See the Limitations section of the paper.

---

## Repository layout

```
├── src/autoencoder/       CAE definitions: autoencoder_32.py, autoencoder_64.py
├── models/                Trained encoder checkpoints (.pkl)
├── data/                  The CAE training corpus
├── logs/                  Per-run TensorBoard logs behind Figures 8 and 9
├── scripts/               Figure generation, checkpoint inspection, log export
├── docs/REPRODUCE.md      Step-by-step reproduction
├── environment.yml        Exact Conda environment used for every result
├── THIRD_PARTY_NOTICES.md Upstream projects this work builds on
└── CITATION.cff
```

---

## Installation

The environment is a faithful export of the machine that produced the results, including the pinned Python 3.7 / PyTorch 1.13.1 CPU stack. Newer versions of `gym` and `stable-baselines3` are **not** drop-in compatible — SB3 1.5.0 requires the pre-0.26 Gym step API.

```bash
conda env create -f environment.yml
conda activate donkeyenv2
```

`gym-donkeycar` is absent from the export because it was installed from source. The experiments used the **`araffin/gym-donkeycar-1` fork**, not the upstream package:

```bash
pip install git+https://github.com/araffin/gym-donkeycar-1.git
```

The fork reports package version 21.02.20, inherited from the upstream release it branched from. You also need the **DonkeySim** simulator binary and must set its path in the Gym environment config before training or evaluating.

---

## Pre-trained encoders

`models/ae-32_minimonaco_600_epochs_best.pkl` and `models/ae-64_minimonaco_600_epochs_best.pkl` are the frozen encoders used for Configs 2–4. They were selected by early stopping on validation MSE (patience 30 epochs, weights restored from the best epoch) after up to 600 epochs of training.

| | Latent dim | Encoder conv layers | Final val. SSIM | Training wall-clock |
|---|---|---|---|---|
| Architecture A | 32 | 4 | 0.625 | 4 h 17 min |
| Architecture B | 64 | 8 | 0.675 | 33 h 47 min |

Both were trained **on CPU only** (Intel Core i5-12500, no discrete GPU), which is why Architecture B took 34 hours. Adam, lr $1\times10^{-3}$, $\beta=(0.9, 0.999)$, batch size 32.

### The `.pkl` files are PyTorch checkpoints

Despite the extension, each file is a serialised checkpoint dictionary rather than a pickled model object:

```
{"state_dict": OrderedDict(...), "data": {"z_size": 32, ...}}
```

`src/autoencoder/autoencoder_32.py` and `autoencoder_64.py` both provide a `load_ae` helper that reads the checkpoint, recovers the latent width from the metadata, and returns a ready `Autoencoder`:

```python
from autoencoder_32 import load_ae

ae = load_ae("models/ae-32_minimonaco_600_epochs_best.pkl")   # z_size recovered from the checkpoint
z  = ae.encode_from_raw_image(frame)      # frame: uint8 HxWx3, RGB
```

For Configs 3 and 4 the agent's state is the latent code with the scalar speed appended, scaled by `v_max = 10`:

```python
import numpy as np
state = np.concatenate([z.flatten(), [min(speed / 10.0, 1.0)]])
```

Input frames are divided by 255 into `[0, 1]` (`preprocess_input(..., mode="rl")`). This must match at inference time, or the latents fall out of distribution and the policy misbehaves for reasons that are hard to trace.

To see what a checkpoint holds before writing code against it:

```bash
python scripts/inspect_checkpoint.py models/ae-32_minimonaco_600_epochs_best.pkl
```

This checkpoint layout is inherited from araffin/aae-train-donkeycar. Note that `torch.load` on PyTorch 2.6 or newer defaults to `weights_only=True` and will reject the non-tensor metadata dictionary.

---

## Dataset

`data/` contains the ~5,000 monocular RGB frames (120×160×3) used to pre-train both autoencoders, with the collection protocol described in `data/README.md`.

This corpus is small by the standards of deep representation learning, and the train/validation split used in the paper was performed **at the frame level**, not by trajectory. Because neighbouring frames are visually similar, the reported reconstruction metrics are optimistic in absolute terms. They are valid for comparing the two architectures with each other, since both were trained on the identical corpus under an identical protocol, but they should not be read as generalisation estimates. This is stated explicitly in the paper.

---

## Reinforcement learning setup

SAC hyperparameters are the **Stable-Baselines3 RL Zoo defaults**, applied unchanged to all four configurations. No per-configuration tuning was performed.

| | |
|---|---|
| `net_arch` | [256, 256] |
| critics | 2 |
| `ent_coef` | auto |
| learning rate | 7.3e-4 |
| $\gamma$ | 0.99 |
| $\tau$ | 0.02 |
| `learning_starts` | 500 |
| `train_freq` | 200 env steps |
| `gradient_steps` | 200 |
| batch size | 256 |
| buffer size | 200,000 |
| total timesteps | 1,000,000 |

**Action space.** Steering $\in [-1,1]$, throttle $\in [0,1]$. The policy emits a squashed Gaussian action in $[-1,1]^2$; the throttle component is rescaled before reaching the simulator, so the car cannot reverse.

**Reward.**

```
r_t = -15 - 2·(v_t / v_max)              if the episode terminates (collision or off-track)
    = -1.0                                if |CTE_t| > 1.5
    = (1 - |CTE_t| / 1.5) · v_t           otherwise
```

CTE is the signed lateral distance between the car's centre and the track centreline, in simulator distance units. The threshold 1.5 is the edge of the *rewarded corridor*, not the off-track boundary: episodes terminate on collision or when `|CTE| > max_cte = 8.0`. No maximum episode length is imposed. The reward is deliberately discontinuous at `|CTE| = 1.5`.

**Evaluation.** Every 10,000 steps, 5 deterministic episodes from the same fixed start pose on the start/finish line. Evaluation transitions never enter the replay buffer. Five independent runs per configuration.

---

## Learning curves and logs

<!-- TODO: this states THREE runs per configuration, which is what logs.zip
     actually contains. The paper states five. If five runs were performed,
     upload the remaining event files and change this number back. If only three
     were performed, correct Section 5.2, Table 3 and the Figure 8 caption in the
     paper to match. These two numbers must agree. -->

`logs/logs.zip` holds the raw TensorBoard event files, three runs per
configuration, laid out as
`<condition>/donkey-minimonaco-track-v0/SAC_<n>/events.out.tfevents.*`. One
`SAC_*` directory is one run.

`scripts/make_figures.py` reads these directly -- no TensorFlow or tbparse
required -- and regenerates every figure in the paper plus a CSV of per-run
aggregate metrics:

```bash
python scripts/make_figures.py --root logs --out ./figures
```

Aggregation follows Agarwal et al. (2021): the interquartile mean across runs,
a min-max envelope as a shaded band, and individual runs drawn faintly behind.
Curves are exponentially smoothed with `SMOOTH_ALPHA = 0.20` and resampled onto
a common step grid before aggregation. `scripts/export_tensorboard.py` will
additionally write a tidy `config, run, step, metric, value` CSV if you want the
curves in plain text.

## Reproducing the paper

See [`docs/REPRODUCE.md`](docs/REPRODUCE.md).

The RL training in this study was run using the two upstream projects below rather than a bespoke trainer, so reproduction routes through them. Both are MIT-licensed; see `THIRD_PARTY_NOTICES.md`.

| Upstream project | Role here |
|---|---|
| [araffin/aae-train-donkeycar](https://github.com/araffin/aae-train-donkeycar) | autoencoder + SAC pipeline for DonkeyCar that this study's two-stage design follows |
| [DLR-RM/rl-baselines3-zoo](https://github.com/DLR-RM/rl-baselines3-zoo) | source of the SAC hyperparameters used unchanged across all four configurations |
| [araffin/gym-donkeycar-1](https://github.com/araffin/gym-donkeycar-1) | the Gym environment actually used for every experiment (a fork of `tawnkramer/gym-donkeycar`) |




## License

Code and documentation: **MIT** (see `LICENSE`).
Image corpus and model weights: **CC BY 4.0** — reuse freely with attribution to the paper.

## Citing

Please cite the *Array* paper; see `CITATION.cff` or the BibTeX in `docs/REPRODUCE.md`.

## Contact

Zakaria Haja — issues and questions are welcome via the GitHub issue tracker.
