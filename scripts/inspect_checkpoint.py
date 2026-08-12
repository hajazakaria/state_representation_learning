"""Print the contents of a CAE checkpoint.

The ``.pkl`` files in ``models/`` are PyTorch checkpoints holding a parameter
dictionary plus training metadata. Run this to see exactly what is inside one
before writing code against it -- particularly the recorded latent width and the
layer names, which tell you whether the file stores the encoder alone or the
full autoencoder.

    python scripts/inspect_checkpoint.py models/ae-32_minimonaco_600_epochs_best.pkl
"""

import argparse
import sys

import torch


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path", help="path to a .pkl / .pt checkpoint")
    ap.add_argument(
        "--all-keys", action="store_true", help="list every parameter tensor"
    )
    args = ap.parse_args()

    try:
        ckpt = torch.load(args.path, map_location="cpu")
    except Exception:
        ckpt = torch.load(args.path, map_location="cpu", weights_only=False)

    print("file        :", args.path)
    print("top level   :", type(ckpt).__name__)

    if not isinstance(ckpt, dict):
        print("\nNot a dict. If this is a full nn.Module, load it directly.")
        return 0

    print("top keys    :", sorted(ckpt.keys()))

    meta = ckpt.get("data")
    if isinstance(meta, dict):
        print("\n-- metadata ('data') --")
        for k in sorted(meta):
            print("   %-24s %r" % (k, meta[k]))

    state = None
    for key in ("state_dict", "model_state_dict", "encoder_state_dict"):
        if key in ckpt:
            state = ckpt[key]
            print("\n-- parameters ('%s') --" % key)
            break
    if state is None and ckpt and all(torch.is_tensor(v) for v in ckpt.values()):
        state = ckpt
        print("\n-- parameters (top level) --")

    if state is None:
        print("\nNo parameter dictionary found.")
        return 1

    keys = list(state.keys())
    total = sum(v.numel() for v in state.values() if torch.is_tensor(v))
    print("   tensors   :", len(keys))
    print("   parameters:", "{:,}".format(total))

    shown = keys if args.all_keys else keys[:6] + (["..."] if len(keys) > 12 else []) + keys[-6:] if len(keys) > 12 else keys
    for k in shown:
        if k == "...":
            print("   ...")
            continue
        v = state[k]
        print("   %-44s %s" % (k, tuple(v.shape) if torch.is_tensor(v) else type(v).__name__))

    # The practical question: encoder only, or the whole autoencoder?
    has_decoder = any(
        tok in k.lower() for k in keys for tok in ("decoder", "deconv", "transpose")
    )
    print(
        "\ncontents    :",
        "full autoencoder (encoder + decoder)" if has_decoder else "encoder only",
    )
    if has_decoder:
        print(
            "note        : use load_ae() from src/autoencoder/autoencoder_32.py,"
            "\n              which instantiates the full model and restores every key."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
