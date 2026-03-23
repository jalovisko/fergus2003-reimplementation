"""
kadir_brady.py
Entropy-based salient region detector.
Kadir & Brady, IJCV 2001.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter
from skimage.util import view_as_windows


def kadir_brady(
    img: np.ndarray,
    n_features: int = 30,
    s_min: int = 4,
    s_max: int = 40,
    s_step: int = 2,
    n_bins: int = 16,
) -> np.ndarray:
    """
    Detect salient regions in a greyscale image.

    Parameters
    ----------
    img        : (H, W) float image, values in [0, 1].
    n_features : maximum number of features to return.
    s_min/max  : scale range (radius in pixels).
    s_step     : scale stride.
    n_bins     : intensity histogram bins.

    Returns
    -------
    features : (N, 4) array, [x, y, scale, saliency], sorted by saliency desc.
    """
    img = np.asarray(img, dtype=np.float64)
    if img.ndim == 3:
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    H, W = img.shape
    img_int = (img * (n_bins - 1)).astype(np.int32).clip(0, n_bins - 1)

    scales = range(s_min, s_max + 1, s_step)

    best_saliency = np.zeros((H, W), dtype=np.float64)
    best_scale    = np.zeros((H, W), dtype=np.float64)
    prev_entropy  = None

    for s in scales:
        entropy = _entropy_map(img_int, s, n_bins, H, W)

        if prev_entropy is None:
            dH = np.zeros_like(entropy)
        else:
            dH = np.abs(entropy - prev_entropy)

        saliency = entropy * dH

        better = saliency > best_saliency
        best_saliency[better] = saliency[better]
        best_scale[better]    = s
        prev_entropy = entropy

    # Zero out border
    border = s_max + 1
    best_saliency[:border, :]  = 0
    best_saliency[-border:, :] = 0
    best_saliency[:, :border]  = 0
    best_saliency[:, -border:] = 0

    # Non-maximum suppression
    nms = maximum_filter(best_saliency, size=9)
    local_max = (best_saliency == nms) & (best_saliency > 0)

    ys, xs = np.where(local_max)
    sals   = best_saliency[ys, xs]
    scls   = best_scale[ys, xs]

    order = np.argsort(-sals)
    ys, xs, sals, scls = ys[order], xs[order], sals[order], scls[order]

    n = min(n_features, len(ys))
    features = np.stack([xs[:n], ys[:n], scls[:n], sals[:n]], axis=1).astype(np.float64)
    return features


def _entropy_map(img_int: np.ndarray, s: int, n_bins: int, H: int, W: int) -> np.ndarray:
    """Approximate per-pixel Shannon entropy using box-filter histograms."""
    # Build per-pixel histograms using integral images for each bin
    # One integral image per bin value
    entropy = np.zeros((H, W), dtype=np.float64)
    integral = np.zeros((H + 1, W + 1), dtype=np.float64)

    bin_integrals = []
    for b in range(n_bins):
        mask = (img_int == b).astype(np.float64)
        # Compute 2-D prefix sum
        ii = np.zeros((H + 1, W + 1), dtype=np.float64)
        ii[1:, 1:] = np.cumsum(np.cumsum(mask, axis=0), axis=1)
        bin_integrals.append(ii)

    for r in range(H):
        r1 = max(r - s, 0);  r2 = min(r + s + 1, H)
        for c in range(W):
            c1 = max(c - s, 0);  c2 = min(c + s + 1, W)
            area = (r2 - r1) * (c2 - c1)
            if area == 0:
                continue
            counts = np.array([
                bin_integrals[b][r2, c2]
                - bin_integrals[b][r1, c2]
                - bin_integrals[b][r2, c1]
                + bin_integrals[b][r1, c1]
                for b in range(n_bins)
            ])
            p = counts / area
            p = p[p > 0]
            entropy[r, c] = -np.sum(p * np.log2(p + 1e-300))

    return entropy
