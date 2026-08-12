"""Load a frozen CAE encoder and turn camera frames into latent state vectors.

The bridge between the pre-trained autoencoders in ``models/`` and the state
vectors consumed by the SAC agent in Configs 2-4 of the paper:

    Config 2:  s_t = z_t                    (32-dim)
    Config 3:  s_t = [z_t, v_t]             (32 + 1)
    Config 4:  s_t = [z_t, v_t]             (64 + 1)

where ``v_t`` is the forward speed reported by the simulator, scaled to [0, 1]
by division by ``V_MAX``.

Checkpoint format
-----------------
The ``.pkl`` files are serialised PyTorch checkpoints, not plain pickled
objects. Each holds a parameter dictionary alongside training metadata::

    {"state_dict": OrderedDict(...), "data": {"z_size": 32, ...}}

This is the save format of araffin/aae-train-donkeycar, on which this pipeline
is based. Loading is therefore three steps: read the checkpoint, use the
metadata to instantiate the right architecture, then load the tensors into it.
Run ``scripts/inspect_checkpoint.py`` to print what a given file contains.

Part of the code and data release for:
  Haja et al., "The Impact of State Representation on Entropic Control for
  Autonomous Racing", Array (under review), 2026.

Licensed MIT. The two-stage encoder/policy design and this checkpoint layout
follow araffin/aae-train-donkeycar (MIT); see THIRD_PARTY_NOTICES.md.
"""

import numpy as np
import torch

# Normalising constant for the proprioceptive channel. Matches v_max in the
# reward function (Equation 5), in simulator velocity units -- the simulator
# does not expose a physical unit for speed.
V_MAX = 10.0

# Simulator camera resolution. Frames arrive as uint8 HxWxC.
FRAME_SHAPE = (120, 160, 3)

# Metadata keys that may carry the latent width, in order of preference.
_Z_KEYS = ("z_size", "latent_dim", "z_dim", "n_latent")


def _build_model(z_size):
    """Instantiate the architecture matching a latent width.

    Architecture A (32-dim, 4 conv layers) is paper Figure 4; Architecture B
    (64-dim, 8 conv layers) is Figure 5. Both live in this package.
    """
    if z_size == 32:
        from .cae_32 import CAE32 as Model
    elif z_size == 64:
        from .cae_64 import CAE64 as Model
    else:
        raise ValueError(
            "no architecture registered for z_size=%r; this study used 32 and 64"
            % (z_size,)
        )
    try:
        return Model(z_size=z_size)
    except TypeError:
        # Architectures with the latent width hard-coded rather than a kwarg.
        return Model()


def load_checkpoint(path):
    """Read a checkpoint file and return ``(state_dict, metadata)``."""
    # map_location keeps this working anywhere; every result in the paper was
    # produced on CPU.
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception:
        # PyTorch >= 2.6 defaults to weights_only=True, which rejects the
        # non-tensor metadata dict. Retry explicitly.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)

    if not isinstance(ckpt, dict):
        raise TypeError(
            "expected a checkpoint dict, got %s -- run "
            "scripts/inspect_checkpoint.py on this file" % type(ckpt).__name__
        )

    state = None
    for key in ("state_dict", "model_state_dict", "encoder_state_dict"):
        if key in ckpt:
            state = ckpt[key]
            break
    if state is None:
        # Some checkpoints are the state_dict itself, tensors at top level.
        if ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
            state = ckpt
        else:
            raise KeyError(
                "no state_dict found; top-level keys are %s" % sorted(ckpt.keys())
            )

    meta = ckpt.get("data") or {}
    return state, meta


def load_encoder(path, z_size=None, strict=True):
    """Load a frozen encoder, ready for inference.

    Parameters
    ----------
    path : str
        ``models/cae_32.pkl`` or ``models/cae_64.pkl``.
    z_size : int, optional
        Latent width. Read from the checkpoint metadata when omitted; pass it
        explicitly if the metadata does not record it.
    strict : bool
        Forwarded to ``load_state_dict``. Set ``False`` when the checkpoint
        stores the full autoencoder but you are instantiating the encoder alone.
        Inspect the reported unmatched keys before trusting the result: silently
        skipping a layer yields latents that look plausible and are wrong.

    Returns
    -------
    torch.nn.Module
        In ``eval`` mode with gradients disabled.
    """
    state, meta = load_checkpoint(path)

    if z_size is None:
        for key in _Z_KEYS:
            if key in meta:
                z_size = int(meta[key])
                break
    if z_size is None:
        raise ValueError(
            "latent width not recorded in checkpoint metadata (keys present: "
            "%s); pass z_size=32 or z_size=64 explicitly" % sorted(meta.keys())
        )

    model = _build_model(z_size)
    result = model.load_state_dict(state, strict=strict)
    if not strict:
        missing = getattr(result, "missing_keys", [])
        unexpected = getattr(result, "unexpected_keys", [])
        if missing:
            print("[load_encoder] missing keys:", list(missing)[:8])
        if unexpected:
            print("[load_encoder] unexpected keys:", list(unexpected)[:8])

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.z_size = z_size
    return model


def preprocess(frame):
    """uint8 HxWxC frame -> float32 1xCxHxW tensor scaled to [0, 1].

    This scaling must match what CAE training used, or the latents fall out of
    distribution and the policy misbehaves for reasons that are hard to trace.
    """
    frame = np.asarray(frame)
    if frame.shape != FRAME_SHAPE:
        raise ValueError(
            "expected a frame of shape %s, got %s" % (FRAME_SHAPE, frame.shape)
        )
    x = frame.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))
    return torch.from_numpy(x).unsqueeze(0)


@torch.no_grad()
def encode(model, frame):
    """Frame -> latent vector z_t as a 1-D float32 numpy array."""
    fn = getattr(model, "encode", model)
    z = fn(preprocess(frame))
    if isinstance(z, (tuple, list)):  # architectures returning (z, ...)
        z = z[0]
    return z.reshape(-1).cpu().numpy().astype(np.float32)


def build_state(z, speed=None):
    """Assemble the state vector the SAC agent receives.

    ``speed=None`` gives Config 2's vision-only state; passing a speed gives the
    augmented state of Configs 3 and 4, scaled to [0, 1].

    The scaled speed is clipped, because the simulator can briefly report a
    speed above V_MAX, which would push one input feature outside the range the
    policy was trained on.
    """
    z = np.asarray(z, dtype=np.float32).reshape(-1)
    if speed is None:
        return z
    v = np.clip(float(speed) / V_MAX, 0.0, 1.0)
    return np.concatenate([z, np.array([v], dtype=np.float32)])


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "models/cae_32.pkl"
    enc = load_encoder(path)
    z = encode(enc, np.zeros(FRAME_SHAPE, dtype=np.uint8))
    print("loaded     :", path)
    print("latent dim :", z.shape[0])
    print("Config 2   :", build_state(z).shape)
    print("Config 3/4 :", build_state(z, speed=4.2).shape)
