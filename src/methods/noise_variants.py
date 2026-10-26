"""The five candidate rung-1 score maps: A-D sensor-noise cues, E = ELA (the winner)."""
import numpy as np
from PIL import Image
from scipy import ndimage

W = 32                 # default local window, pixels
_SQRT_PI_2 = 1.2533    # for Gaussian noise, std = mean(|x|) * sqrt(pi/2)


# ---- Helpers ----

def _highpass(ch):
    """2x2 diagonal high-pass filter: removes smooth image content, keeps fine noise."""
    k = np.array([[1.0, -1.0], [-1.0, 1.0]]) / 2.0
    return ndimage.convolve(ch, k, mode="reflect")


def _local_sigma(r, w=W):
    """Local noise level from the mean absolute residual (less sensitive to edges than std)."""
    return ndimage.uniform_filter(np.abs(r), size=w) * _SQRT_PI_2


def _robust_z(x):
    """How far each value is from the image median, in robust standard deviations."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return np.abs(x - med) / (1.4826 * mad + 1e-8)


# ---- Sensor-noise cues (A-D) ----

def v_median(img, w=W):
    """A: median-filter residual, local standard deviation."""
    x = np.asarray(img, float).mean(2) / 255.0          # greyscale in [0, 1]
    r = x - ndimage.median_filter(x, size=3)            # residual = image - denoised
    m1 = ndimage.uniform_filter(r, size=w)
    m2 = ndimage.uniform_filter(r ** 2, size=w)
    return _robust_z(np.sqrt(np.maximum(m2 - m1 ** 2, 0)))


def v_highpass(img, w=W):
    """B: high-pass residual, robust local noise level."""
    x = np.asarray(img, float).mean(2) / 255.0
    return _robust_z(_local_sigma(_highpass(x), w))


def v_ratio(img, w=W):
    """C: noise divided by local texture, to cancel out busy scene content."""
    x = np.asarray(img, float).mean(2) / 255.0
    sigma = _local_sigma(_highpass(x), w)                         # local noise
    gx, gy = ndimage.sobel(x, 0), ndimage.sobel(x, 1)             # image gradients
    texture = ndimage.uniform_filter(np.hypot(gx, gy), size=w)    # local edge strength
    return _robust_z(sigma / (texture + 1e-3))


def v_ratio_colour(img, w=W):
    """D: as C, but computed per colour channel and then averaged."""
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


# ---- Compression cue (E) ----

def v_ela(img, quality=90, w=32):
    """E: Error Level Analysis (rung 1): how much each pixel changes when re-saved as JPEG."""
    from io import BytesIO
    buf = BytesIO()
    Image.fromarray(np.asarray(img, np.uint8)).save(buf, "JPEG", quality=quality)  # re-save in memory
    buf.seek(0)
    re_ = np.asarray(Image.open(buf).convert("RGB"), float)                       # the re-saved image
    diff = np.abs(np.asarray(img, float) - re_).mean(2)                           # per-pixel change
    return _robust_z(ndimage.uniform_filter(diff, size=w))                        # smooth w x w, standardise


VARIANTS["E_ela"] = v_ela
