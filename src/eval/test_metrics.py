import numpy as np
from metrics import f1_iou, image_auc

t = np.zeros((10, 10), bool); t[2:8, 2:8] = True     # 36 true pixels

# identical
r = f1_iou(t, t);                        assert r["f1"] == 1.0 and r["iou"] == 1.0

# disjoint
d = np.zeros((10, 10), bool); d[0:2, 0:2] = True
r = f1_iou(d, t);                        assert r["f1"] == 0.0 and r["iou"] == 0.0

# prediction covers exactly half the true region, no false positives
h = np.zeros((10, 10), bool); h[2:5, 2:8] = True     # 18 pixels, all inside t
r = f1_iou(h, t)
assert abs(r["f1"] - 2/3) < 1e-9, r["f1"]            # 2*18 / (2*18 + 0 + 18)
assert abs(r["iou"] - 0.5) < 1e-9, r["iou"]

# both empty -> 1.0 by our stated convention
e = np.zeros((10, 10), bool)
r = f1_iou(e, e);                        assert r["f1"] == 1.0

# one empty -> 0.0
r = f1_iou(e, t);                        assert r["f1"] == 0.0
r = f1_iou(t, e);                        assert r["f1"] == 0.0

# AUC: perfect separation, reversed, and pure ties
assert image_auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
assert image_auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
assert image_auc([0.5, 0.5, 0.5, 0.5], [0, 0, 1, 1]) == 0.5

# pixel_auc: ties take the average rank, so a constant map scores exactly 0.5
from metrics import pixel_auc
_t = np.zeros((40, 40), bool); _t[10:20, 10:20] = True
assert abs(pixel_auc(np.ones((40, 40)), _t) - 0.5) < 1e-12
assert abs(pixel_auc(_t.astype(float), _t) - 1.0) < 1e-12
assert pixel_auc(np.ones((40, 40)), np.zeros((40, 40), bool)) is None

print("all metric tests pass")
