# Third-party notices

This work builds on the projects below. All are distributed under the MIT
License, which requires their copyright notice and permission notice to be
retained in any derived work. Where code in this repository is derived from or
adapted from these projects, that origin is noted in the file header.

---

## araffin/aae-train-donkeycar

<https://github.com/araffin/aae-train-donkeycar>
Copyright (c) 2021 Antonin Raffin — MIT License

Training an autoencoder on DonkeyCar camera frames and then training an RL agent
on the resulting latent codes. The two-stage design of the present study —
offline representation learning followed by policy learning on a frozen encoder
— follows this project, as does the checkpoint layout of the models in
`models/`: a `state_dict` of parameter tensors alongside a `data` dictionary of
training metadata.

## DLR-RM/rl-baselines3-zoo

<https://github.com/DLR-RM/rl-baselines3-zoo>
Copyright (c) 2019 Antonin Raffin — MIT License

A training framework and hyperparameter collection for Stable-Baselines3. The
SAC hyperparameters in Table 2 of the paper are this project's defaults, applied
unchanged to all four configurations — including the learning rate of 7.3e-4,
`train_freq` and `gradient_steps` of 200, `learning_starts` of 500, and
`ent_coef = auto`. Because no per-configuration search was performed, no
configuration, including the end-to-end baseline, was advantaged or
disadvantaged by tuning.

## araffin/gym-donkeycar-1

<https://github.com/araffin/gym-donkeycar-1>
MIT License — a fork of `tawnkramer/gym-donkeycar`
Copyright (c) 2019 Tawn Kramer and contributors

**This fork, at package version 21.02.20, is the environment used for every
experiment in the paper**, not the upstream package. The Gym interface to the
DonkeyCar simulator, providing the MiniMonaco circuit.

For exact reproducibility, record the commit and the simulator build:

```
gym-donkeycar-1 commit: __________________________
DonkeySim build:        __________________________
```

The commit matters because a fork's package version string is inherited from the
release it forked from and does not change as the fork advances, so `21.02.20`
alone does not identify the code that ran.

---

## Stable-Baselines3

<https://github.com/DLR-RM/stable-baselines3> — MIT License
Version 1.5.0. Provides the SAC implementation, `MlpPolicy`, and the `CnnPolicy`
whose NatureCNN feature extractor is the Config 1 baseline.

## Other libraries

PyTorch (BSD-3-Clause), NumPy (BSD-3-Clause), OpenCV (Apache-2.0),
scikit-image (BSD-3-Clause, source of the SSIM implementation), Matplotlib
(PSF-based license), Optuna (MIT), TensorBoard (Apache-2.0). Exact versions are
pinned in `environment.yml`.
