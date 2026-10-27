"""Checks on the rung 4/5 network (CPU, a few seconds): shapes, decoder, constraint, sidecars, overfitting.

Run from the repo root:  python3 src/methods/test_unet.py
"""
import json, os, sys, tempfile                   # JSON files, paths, temporary folder
import torch                                      # deep-learning library

sys.path.insert(0, os.path.dirname(__file__))     # so files in this folder can be imported
from unet import UNet, up, dice_bce               # the model, its upsampling block, its loss
import refine_meta                                # the config sidecars

torch.manual_seed(0)                              # same random numbers every run


# ---- 1. Output is the same size as the input, for both decoders ----

def test_shapes():
    for scheme in ("resize", "transpose"):
        m = UNet(upsample=scheme).eval()          # evaluation mode
        for h, w in ((64, 64), (72, 136)):        # sizes that are multiples of 8
            with torch.no_grad():                 # no gradients needed
                y = m(torch.rand(1, 3, h, w))     # one random image
            assert y.shape == (1, 1, h, w), (scheme, y.shape)   # one logit per pixel
    print("shapes ok")


# ---- 2. The default decoder has no transposed convolutions ----

def test_default_has_no_transpose():
    m = UNet()                                    # default settings
    assert m.upsample == "resize"
    assert not any(isinstance(x, torch.nn.ConvTranspose2d) for x in m.modules())   # none anywhere in the model
    n_new = sum(p.numel() for p in m.parameters())                       # parameter count, new decoder
    n_old = sum(p.numel() for p in UNet(upsample="transpose").parameters())   # and old
    print(f"no transposed convs; parameters {n_new/1e6:.3f} M (was {n_old/1e6:.3f} M)")


# ---- 3. The point of the switch: no checkerboard from a constant input ----

def test_no_periodic_pattern():
    x = torch.ones(1, 16, 8, 8)                   # a constant feature map
    for scheme, expect_flat in (("resize", True), ("transpose", False)):
        u = up(16, 4, scheme)                     # one upsampling block, random weights
        with torch.no_grad():
            y = u(x)[0, :, 2:-2, 2:-2]                 # interior: no padding effects
        # spread of each channel over the four positions of the 2x2 phase grid
        phases = torch.stack([y[:, i::2, j::2].mean(dim=(1, 2))
                              for i in (0, 1) for j in (0, 1)])   # mean at each of the 4 positions
        spread = (phases.max(0).values - phases.min(0).values).max().item()   # biggest difference
        flat = spread < 1e-5                      # flat = all four positions the same
        assert flat == expect_flat, (scheme, spread)   # resize: flat; transposed: a period-2 pattern
        print(f"{scheme:<9} constant in -> phase spread {spread:.2e} "
              f"({'flat' if flat else 'period-2 pattern'})")


# ---- 4. The constrained first layer still obeys its rule ----

def test_constraint():
    m = UNet()
    m.pre.weight.data.normal_()                   # scramble the first layer's weights
    m.constrain()                                 # re-apply the constraint
    w = m.pre.weight.data
    c = m.pre.k // 2                              # index of the centre weight
    assert torch.allclose(w[:, :, c, c], torch.full_like(w[:, :, c, c], -1.0))   # centre = -1
    surround = w.sum(dim=(2, 3)) - w[:, :, c, c]  # sum of the other weights
    assert torch.allclose(surround, torch.ones_like(surround), atol=1e-5)        # surround sums to 1
    print("constrained layer ok")


# ---- 5. A sidecar from before the decoder switch is refused ----

def test_old_sidecar_refused():
    cfg = {k: 0 for k in refine_meta.KEYS}        # a dummy config with every required key
    cfg["upsample"] = "resize"
    with tempfile.TemporaryDirectory() as d:      # a scratch folder, deleted afterwards
        p = os.path.join(d, "m.pt")
        refine_meta.write(p, cfg)                 # write the sidecar
        refine_meta.check(p, cfg)                      # same config: accepted
        old = {k: v for k, v in cfg.items() if k != "upsample"}   # the same config without "upsample"
        json.dump(old, open(refine_meta.sidecar(p), "w"))         # overwrite the sidecar with it
        try:
            refine_meta.check(p, cfg)             # must stop the program ...
        except SystemExit:
            print("pre-switch sidecar refused")
        else:                                     # ... so getting here is a failure
            raise AssertionError("a sidecar without 'upsample' was accepted")


# ---- 6. The network can learn one batch (the wiring is right) ----

def test_overfit():
    m = UNet()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)   # optimiser
    x = torch.rand(2, 3, 64, 64)                  # two random images
    y = torch.zeros(2, 1, 64, 64)                 # their masks: empty ...
    y[:, :, 16:40, 20:44] = 1                     # ... except one rectangle
    m.constrain()
    first = None                                  # loss at the first step
    for _ in range(60):                           # 60 training steps on the same batch
        opt.zero_grad()
        loss = dice_bce(m(x), y)
        loss.backward()
        opt.step()
        m.constrain()                             # keep the first layer constrained
        first = first if first is not None else loss.item()
    assert loss.item() < 0.5 * first, (first, loss.item())   # loss must at least halve
    print(f"overfits one batch: loss {first:.3f} -> {loss.item():.3f}")


if __name__ == "__main__":
    test_shapes()
    test_default_has_no_transpose()
    test_no_periodic_pattern()
    test_constraint()
    test_old_sidecar_refused()
    test_overfit()
    print("test_unet: all passed")
