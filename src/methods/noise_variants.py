"""Candidate rung-1 score maps. All classical, no learned parameters."""
import numpy as np
from PIL import Image
from scipy import ndimage

W = 32
_SQRT_PI_2 = 1.2533  # E|X| -> sigma for Gaussian X


def _highpass(ch):
    """Diagonal 2x2 high-pass. Suppresses smooth structure; passes noise."""
    k = np.array([[1.0, -1.0], [-1.0, 1.0]]) / 2.0
    return ndimage.convolve(ch, k, mode="reflect")


def _local_sigma(r, w=W):
    """Local noise scale via mean-absolute residual (robust to edge outliers)."""
    return ndimage.uniform_filter(np.abs(r), size=w) * _SQRT_PI_2


def _robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return np.abs(x - med) / (1.4826 * mad + 1e-8)


def v_median(img, w=W):
    """A: original -- median-filter residual, local std."""
    x = np.asarray(img, float).mean(2) / 255.0
    r = x - ndimage.median_filter(x, size=3)
    m1 = ndimage.uniform_filter(r, size=w)
    m2 = ndimage.uniform_filter(r ** 2, size=w)
    return _robust_z(np.sqrt(np.maximum(m2 - m1 ** 2, 0)))


def v_highpass(img, w=W):
    """B: diagonal high-pass, robust local sigma."""
    x = np.asarray(img, float).mean(2) / 255.0
    return _robust_z(_local_sigma(_highpass(x), w))


def v_ratio(img, w=W):
    """C: noise-to-texture ratio -- the key idea. Divides out scene content."""
    x = np.asarray(img, float).mean(2) / 255.0
    sigma = _local_sigma(_highpass(x), w)
    gx, gy = ndimage.sobel(x, 0), ndimage.sobel(x, 1)
    texture = ndimage.uniform_filter(np.hypot(gx, gy), size=w)
    return _robust_z(sigma / (texture + 1e-3))


def v_ratio_colour(img, w=W):
    """D: as C, but per-channel then averaged -- demosaicing differs per channel."""
    x = np.asarray(img, float) / 255.0
    out = []
    for c in range(x.shape[2]):
        ch = x[..., c]
        sigma = _local_sigma(_highpass(ch), w)
        gx, gy = ndimage.sobel(ch, 0), ndimage.sobel(ch, 1)
        texture = ndimage.uniform_filter(np.hypot(gx, gy), size=w)
        out.append(sigma / (texture + 1e-3))
    return _robust_z(np.mean(out, axis=0))


VARIANTS = {"A_median": v_median, "B_highpass": v_highpass,
            "C_ratio": v_ratio, "D_ratio_colour": v_ratio_colour}


def v_ela(img, quality=90, w=16):
    """E: Error Level Analysis. Re-compress at fixed quality and measure the
    error. A region with a different compression history responds differently
    from its host -- targets compression, not sensor noise."""
    from io import BytesIO
    buf = BytesIO()
    Image.fromarray(np.asarray(img, np.uint8)).save(buf, "JPEG", quality=quality)
    buf.seek(0)
    re_ = np.asarray(Image.open(buf).convert("RGB"), float)
    diff = np.abs(np.asarray(img, float) - re_).mean(2)
    return _robust_z(ndimage.uniform_filter(diff, size=w))


VARIANTS["E_ela"] = v_ela
