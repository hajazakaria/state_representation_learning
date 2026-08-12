# CAE pre-training corpus

Approximately **5,000 monocular RGB frames**, 120×160×3, from the DonkeyCar
MiniMonaco circuit. This is the corpus both autoencoders in `models/` were
trained on.


## Collection protocol

1. Continuous video was recorded while the vehicle was **driven manually** through
   complete laps of the MiniMonaco circuit.
2. The recording was decoded into individual frames.
3. Approximately 5,000 frames were **selected manually**, chosen to cover the full
   circuit and to avoid over-representing any single section of track.

Frames span the lighting and track conditions present in the MiniMonaco
environment. Because selection was manual rather than uniform-in-time, the
corpus is not a uniform sample of the state distribution induced by any
particular policy.

## Preprocessing at training time

- Pixel values scaled to [0, 1].
- 80 / 20 train / validation split.

> **The split is frame-level, not trajectory-level.** Frames were assigned to the
> two sets independently, so temporally adjacent — and therefore visually very
> similar — frames appear on both sides of the split. The reconstruction metrics
> reported in the paper (validation MSE, SSIM) are consequently **optimistic** as
> estimates of reconstruction quality on genuinely unseen viewpoints.
>
> They remain valid for the comparison the paper actually makes, between
> Architecture A and Architecture B, since both were trained on the identical
> corpus under the identical protocol. They should not be quoted as
> generalisation figures. A trajectory-level split — holding out whole laps —
> would be the cleaner design and is identified as future work.

## Size caveat

Around 5,000 images is small for deep representation learning, and the paper
says so explicitly. Two things bound the risk of overfitting for the study's
purposes: early stopping on validation MSE with a patience of 30 epochs, with
weights restored from the best epoch; and the fact that both architectures faced
the identical corpus, so corpus size does not confound the comparison between
them. What it does limit is the absolute generalisability of either latent space.

## License

**CC BY 4.0.** Reuse freely with attribution to the paper (see `CITATION.cff`).
