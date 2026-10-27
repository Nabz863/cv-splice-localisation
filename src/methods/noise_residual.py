"""Sensor-noise splice cue: flag regions whose local noise level is unusual for the image."""
import numpy as np   # arrays and maths
from scipy import ndimage   # image filters

DENOISE_KSIZE = 3      # median filter size; median keeps edges, so the residual is mostly noise
WINDOW = 32            # size (pixels) of the window used to measure local noise


def score_map(img, denoise_ksize=DENOISE_KSIZE, window=WINDOW):
    """Per-pixel score: how unusual the local noise level is (a robust z-score)."""
    x = np.asarray(img, np.float64) / 255.0             # scale pixels to [0, 1]
    if x.ndim == 2:   # a greyscale image has no channel dimension ...
        x = x[:, :, None]                               # treat greyscale as one channel

    sigmas = []   # one noise-level map per colour channel
    for c in range(x.shape[2]):   # for each channel
        ch = x[..., c]   # this channel
        residual = ch - ndimage.median_filter(ch, size=denoise_ksize)    # image minus denoised image
        m1 = ndimage.uniform_filter(residual, size=window)               # local mean of residual
        m2 = ndimage.uniform_filter(residual ** 2, size=window)          # local mean of residual^2
        sigmas.append(np.sqrt(np.maximum(m2 - m1 ** 2, 0.0)))           # local std = noise level

    sigma = np.mean(sigmas, axis=0)                     # average over colour channels

    # Median and MAD instead of mean and std, so the spliced region itself (the
    # outlier we are looking for) doesn't distort the image's baseline.
    med = np.median(sigma)   # the image's typical noise level
    mad = np.median(np.abs(sigma - med))   # typical spread around it
    return np.abs(sigma - med) / (1.4826 * mad + 1e-8) # 1.4826*MAD ~ std for Gaussian data


def predict(img, tau, **kw):
    """Binary mask: pixels whose score reaches tau."""
    return score_map(img, **kw) >= tau   # True where the score reaches the threshold
