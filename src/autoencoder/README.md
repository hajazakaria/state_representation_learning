# CAE architectures

| File | Latent | Encoder | Weights |
|---|---|---|---|
| `cae_32.py` | 32 | 4 conv layers | `models/cae_32.pkl` |
| `cae_64.py` | 64 | 8 conv layers | `models/cae_64.pkl` |
| `load_encoder.py` | -- | loads either, builds the state vector | -- |

Place your architecture definitions here as `cae_32.py` and `cae_64.py`. Both
should expose a class with an `encode`/`forward` returning the latent vector, so
`load_encoder.py` works against either unchanged.

Add a header to each file noting that the design follows
araffin/aae-train-donkeycar (MIT) where it does -- see `THIRD_PARTY_NOTICES.md`.
MIT requires the upstream notice to be retained in derived work.

## Architecture A -- 32-dim (paper Figure 4)

Encoder: Conv(3x3, s=1, 32) - Conv(4x4, s=2, 32) - Conv(3x3, s=1, 64) -
Conv(4x4, s=2, 64), ReLU throughout, then Flatten - FC -> z in R^32.
Decoder mirrors it with ConvTranspose, Sigmoid output, 120x160x3.

## Architecture B -- 64-dim (paper Figure 5)

Encoder: as above, continuing to 128 and 256 channels for 8 conv layers total,
then Flatten - FC -> z in R^64. Decoder mirrors it.

Inputs are scaled to [0, 1]. This must match at inference time or the latents go
out of distribution.
