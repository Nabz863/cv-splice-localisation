"""Rung 1: noise-residual splice detector. No learned parameters.

A camera's noise statistics are a property of the capture chain, not the scene.
A spliced region therefore carries a foreign noise floor. We estimate the local
noise level everywhere and flag pixels whose level is anomalous *relative to
this image's own* floor -- absolute noise level varies hugely across cameras
and JPEG qualities, so only the within-image contrast is meaningful.
"""
import numpy as np
from scipy import ndimage

DENOISE_KSIZE = 3      # median filter: edge-preserving, so residual is noise not structure
WINDOW = 32            # local-variance window, in pixels


def score_map(img, denoise_ksize=DENOISE_KSIZE, window=WINDOW):
    """Dense evidence map. Higher = more anomalous noise level.

    Returns a robust z-score, so it is comparable across images and a single
    threshold tau is meaningful for the whole dataset.
    """
    x = np.asarray(img, np.float64) / 255.0
    if x.ndim == 2:
        x = x[:, :, None]

    sigmas = []
    for c in range(x.shape[2]):
        ch = x[..., c]
        residual = ch - ndimage.median_filter(ch, size=denoise_ksize)
        m1 = ndimage.uniform_filter(residual, size=window)
        m2 = ndimage.uniform_filter(residual ** 2, size=window)
        sigmas.append(np.sqrt(np.maximum(m2 - m1 ** 2, 0.0)))

    sigma = np.mean(sigmas, axis=0)

    # Robust centre/scale: median and MAD, not mean/std, because the tampered
    # region is exactly the outlier we must not let inflate our own baseline.
    med = np.median(sigma)
    mad = np.median(np.abs(sigma - med))
    return np.abs(sigma - med) / (1.4826 * mad + 1e-8)


def predict(img, tau, **kw):
    """Binary mask at threshold tau."""
    return score_map(img, **kw) >= tau
