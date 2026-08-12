# Trained models

## Frozen CAE encoders

| File | Latent dim | Used by | Encoder conv layers | Final val. SSIM | Training wall-clock |
|---|---|---|---|---|---|
| `ae-32_minimonaco_600_epochs_best.pkl` | 32 | Configs 2 and 3 | 4 | ~0.625 | 4 h 17 min |
| `ae-64_minimonaco_600_epochs_best.pkl` | 64 | Config 4 | 8 | ~0.675 | 33 h 47 min |

Both were trained for up to 600 epochs with Adam (lr 1e-3, beta = (0.9, 0.999)),
batch size 32, on the corpus in `data/`. Early stopping monitored validation MSE
with a patience of 30 epochs, so the saved weights are from the epoch with the
lowest validation loss, not the final epoch.

Training was **CPU-only** on an Intel Core i5-12500 with no discrete GPU, which
is why the 64-dim model took nearly 34 hours. Architecture definitions are in
`src/autoencoder/autoencoder_32.py` and `autoencoder_64.py`.

During RL training these encoders are **frozen**: used purely as a fixed
function from observation to latent vector, receiving no gradient from the RL
objective.

### Architectures

**Architecture A (32-dim, paper Figure 4).** Four convolutional layers, kernel
4x4, stride 2, channels 16 -> 32 -> 64 -> 128, each followed by ReLU; flatten;
fully connected layer to z in R^32. The decoder mirrors this with transposed
convolutions.

**Architecture B (64-dim, paper Figure 5).** Eight convolutional layers in
alternating blocks -- a 3x3 stride-1 convolution that refines features followed
by a 4x4 stride-2 convolution that halves the resolution -- with channels
progressing 32 -> 64 -> 128 -> 256; flatten; fully connected layer to z in R^64.
The decoder mirrors this.

### Format

Despite the `.pkl` extension these are PyTorch checkpoints, not pickled model
objects. Each contains a parameter dictionary plus training metadata:

```python
{"state_dict": OrderedDict(...), "data": {"z_size": 32, ...}}
```

Use the `load_ae` helper that ships with the architecture files; it recovers the
latent width from the metadata, so you do not have to pass it:

```python
from autoencoder_32 import load_ae
ae = load_ae("models/ae-32_minimonaco_600_epochs_best.pkl")
z  = ae.encode_from_raw_image(frame)
```

To inspect a checkpoint before writing code against it:

```bash
python scripts/inspect_checkpoint.py models/ae-32_minimonaco_600_epochs_best.pkl
```

Two things to know:

- **`weights_only` on modern PyTorch.** From 2.6 onward `torch.load` defaults to
  `weights_only=True`, which refuses the non-tensor metadata dictionary. Pass
  `weights_only=False` if you load a checkpoint by hand.
- **Version sensitivity.** These were written under PyTorch 1.13.1. Because they
  store tensors rather than a serialised class they load across versions far
  more reliably than a pickled module would, but recreate the environment from
  `environment.yml` if you hit a key mismatch.

## SAC policies

Not included. The reinforcement learning artefacts provided here are the
per-run training logs in `logs/`, which are sufficient to regenerate every
figure and table in the paper.
