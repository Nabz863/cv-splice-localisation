"""Checks on the rung 4/5 network, CPU only, a few seconds.

1. Both decoders map an H x W input (H, W multiples of 8) to H x W logits.
2. The default decoder is resize-then-convolve and contains no transposed conv.
3. The point of the switch: fed a constant feature map, the resize decoder's
   upsampling gives a constant output (away from the zero-padded border), while
   a 2x2 stride-2 transposed conv with random weights gives a period-2 pattern,
   because each of the four positions in a 2x2 output block has its own weights.
4. The constrained first layer still satisfies its constraint.
5. refine_meta refuses a sidecar written before the "upsample" key existed.
6. The network can overfit one synthetic batch (the wiring is right).

Run from the repo root:  python3 src/methods/test_unet.py
"""
import json, os, sys, tempfile
import torch

sys.path.insert(0, os.path.dirname(__file__))
from unet import UNet, up, dice_bce
import refine_meta

torch.manual_seed(0)


def test_shapes():
    for scheme in ("resize", "transpose"):
        m = UNet(upsample=scheme).eval()
        for h, w in ((64, 64), (72, 136)):
            with torch.no_grad():
                y = m(torch.rand(1, 3, h, w))
            assert y.shape == (1, 1, h, w), (scheme, y.shape)
    print("shapes ok")


def test_default_has_no_transpose():
    m = UNet()
    assert m.upsample == "resize"
    assert not any(isinstance(x, torch.nn.ConvTranspose2d) for x in m.modules())
    n_new = sum(p.numel() for p in m.parameters())
    n_old = sum(p.numel() for p in UNet(upsample="transpose").parameters())
    print(f"no transposed convs; parameters {n_new/1e6:.3f} M (was {n_old/1e6:.3f} M)")


def test_no_periodic_pattern():
    x = torch.ones(1, 16, 8, 8)
    for scheme, expect_flat in (("resize", True), ("transpose", False)):
        u = up(16, 4, scheme)
        with torch.no_grad():
            y = u(x)[0, :, 2:-2, 2:-2]                 # interior: no padding effects
        # spread of each channel over the four positions of the 2x2 phase grid
        phases = torch.stack([y[:, i::2, j::2].mean(dim=(1, 2))
                              for i in (0, 1) for j in (0, 1)])
        spread = (phases.max(0).values - phases.min(0).values).max().item()
        flat = spread < 1e-5
        assert flat == expect_flat, (scheme, spread)
        print(f"{scheme:<9} constant in -> phase spread {spread:.2e} "
              f"({'flat' if flat else 'period-2 pattern'})")


def test_constraint():
    m = UNet()
    m.pre.weight.data.normal_()
    m.constrain()
    w = m.pre.weight.data
    c = m.pre.k // 2
    assert torch.allclose(w[:, :, c, c], torch.full_like(w[:, :, c, c], -1.0))
    surround = w.sum(dim=(2, 3)) - w[:, :, c, c]
    assert torch.allclose(surround, torch.ones_like(surround), atol=1e-5)
    print("constrained layer ok")


def test_old_sidecar_refused():
    cfg = {k: 0 for k in refine_meta.KEYS}
    cfg["upsample"] = "resize"
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "m.pt")
        refine_meta.write(p, cfg)
        refine_meta.check(p, cfg)                      # same config: accepted
        old = {k: v for k, v in cfg.items() if k != "upsample"}
        json.dump(old, open(refine_meta.sidecar(p), "w"))
        try:
            refine_meta.check(p, cfg)
        except SystemExit:
            print("pre-switch sidecar refused")
        else:
            raise AssertionError("a sidecar without 'upsample' was accepted")


def test_overfit():
    m = UNet()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    x = torch.rand(2, 3, 64, 64)
    y = torch.zeros(2, 1, 64, 64)
    y[:, :, 16:40, 20:44] = 1
    m.constrain()
    first = None
    for _ in range(60):
        opt.zero_grad()
        loss = dice_bce(m(x), y)
        loss.backward()
        opt.step()
        m.constrain()
        first = first if first is not None else loss.item()
    assert loss.item() < 0.5 * first, (first, loss.item())
    print(f"overfits one batch: loss {first:.3f} -> {loss.item():.3f}")


if __name__ == "__main__":
    test_shapes()
    test_default_has_no_transpose()
    test_no_periodic_pattern()
    test_constraint()
    test_old_sidecar_refused()
    test_overfit()
    print("test_unet: all passed")
