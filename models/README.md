# Trained models

## Frozen CAE encoders

| File | Latent dim | Used by | Encoder conv layers | Final val. SSIM | Final val. MSE trend | Training time |
|---|---|---|---|---|---|---|
| `cae_32.pkl` | 32 | Configs 2 and 3 | 4 | ≈ 0.625 | higher error floor | 4 h 17 min |
| `cae_64.pkl` | 64 | Config 4 | 8 | ≈ 0.675 | lower error floor | 33 h 47 min |

Both were trained for up to 600 epochs with Adam (lr 1e-3, β = (0.9, 0.999)),
batch size 32, on the corpus in `data/`. Early stopping monitored validation MSE
with a patience of 30 epochs, and the saved weights are from the epoch with the
lowest validation loss — not the final epoch.

Training was **CPU-only** on an Intel Core i5-12500 with no discrete GPU, which
is why the 64-dim model took nearly 34 hours. Architecture definitions are in
`src/autoencoder/`.

During RL training these encoders are **frozen**: they are used purely as a
fixed function mapping an observation to a latent vector, and receive no
gradient from the RL objective.

### Loading

```python
from src.autoencoder.load_encoder import load_encoder
encoder = load_encoder("models/cae_32.pkl")
```

### Format

Despite the `.pkl` extension these are PyTorch checkpoints, not pickled model
objects. Each contains a parameter dictionary plus training metadata:

```python
{"state_dict": OrderedDict(...), "data": {"z_size": 32, ...}}
```

The layout is inherited from araffin/aae-train-donkeycar. `load_encoder` reads
the latent width from `data` and instantiates the matching architecture, so you
do not have to pass it. To see what a file holds:

```bash
python scripts/inspect_checkpoint.py models/cae_64.pkl
```

Two things to know:

- **`weights_only` on modern PyTorch.** From 2.6 onward `torch.load` defaults to
  `weights_only=True`, which refuses the non-tensor metadata dictionary. The
  loader catches this and retries with `weights_only=False`.
- **Version sensitivity.** The checkpoints were written under PyTorch 1.13.1.
  Because they store tensors rather than a serialised class, they load across
  versions far more reliably than a pickled module would — but recreate the
  environment from `environment.yml` if you hit a key mismatch.

---

