"""The training configuration of the rung 4/5 models, and a check that a saved
model or cached logits were produced under it.

Before this existed, cache_logits.py loaded any results/models/refine_{tag}.pt it
found. Two pairs (t0v1, t0v2) were in fact models left over from an earlier
script, trained at a different batch size from the other 18, and nothing recorded
that. Now every saved model and logits file has a JSON sidecar holding the config
it was produced under, and loading refuses on any mismatch.

No torch import, so CPU-only scripts can use it.
"""
import json, os

KEYS = ("epochs_max", "patience", "batch", "workers", "lr", "weight_decay", "grad_clip",
        "constraint_norm", "amp_dtype", "n_cal", "crops_per_image", "patch")


def sidecar(path):
    return os.path.splitext(path)[0] + ".json"


def write(path, config, **extra):
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
