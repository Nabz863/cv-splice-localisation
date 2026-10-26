"""Save each U-Net's training config next to it (JSON), and refuse files made under a different config."""
import json, os

# config fields that must match for a saved model / logits file to be reused
KEYS = ("epochs_max", "patience", "batch", "workers", "lr", "weight_decay", "grad_clip",
        "constraint_norm", "amp_dtype", "n_cal", "crops_per_image", "patch", "n_train",
        "crop_rng", "nonfinite_counted")


def sidecar(path):
    """model.pt -> model.json"""
    return os.path.splitext(path)[0] + ".json"


def write(path, config, **extra):
    """Write the config (plus any extra fields) next to path."""
    with open(sidecar(path), "w") as f:
        json.dump({**{k: config[k] for k in KEYS}, **extra}, f, indent=1, sort_keys=True)


def check(path, config):
    """Raise SystemExit unless path's sidecar exists and matches config."""
    sc = sidecar(path)
    if not os.path.exists(sc):
        raise SystemExit(f"{path} has no config sidecar ({sc}), so how it was produced is "
                         f"unknown. Delete it and rerun to regenerate it.")
    have = json.load(open(sc))
    diff = {k: (have.get(k), config[k]) for k in KEYS if have.get(k) != config[k]}
    if diff:
        raise SystemExit(f"{path} was produced under a different config "
                         f"(saved, current): {diff}. Delete it and rerun.")
    return have
