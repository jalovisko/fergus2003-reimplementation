"""
feature_extraction.py
Region detection, patch cropping, PCA projection.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, TypeVar

import numpy as np
from kadir_brady import kadir_brady
from PIL import Image
from sklearn.decomposition import PCA
from tqdm import tqdm

T = TypeVar("T")


def _maybe_tqdm(
    it: Iterable[T], *, desc: str, unit: str, enabled: bool
) -> Iterable[T]:
    if not enabled:
        return it
    return tqdm(it, desc=desc, unit=unit, dynamic_ncols=True)


@dataclass
class Features:
    """Per-image features."""
    X: np.ndarray   # (N, 2)  pixel locations [x, y]
    S: np.ndarray   # (N,)    log-scale
    A: np.ndarray   # (N, K)  PCA-projected appearance


class FeatureExtractor:
    """
    Extracts salient-region features and projects patches via PCA.

    Parameters
    ----------
    n_feat    : max features per image
    patch_sz  : patch side length in pixels (square)
    k_pca     : PCA dimensions to keep
    s_min/max : scale range for detector
    """

    def __init__(
        self,
        n_feat:   int = 30,
        patch_sz: int = 11,
        k_pca:    int = 15,
        s_min:    int = 4,
        s_max:    int = 40,
        progress: bool = True,
    ):
        self.n_feat   = n_feat
        self.patch_sz = patch_sz
        self.k_pca    = k_pca
        self.s_min    = s_min
        self.s_max    = s_max
        self.progress = progress
        self.pca: Optional[PCA] = None

    # ------------------------------------------------------------------
    def fit(self, images: List[np.ndarray]) -> "FeatureExtractor":
        """Fit PCA on all patches from images. Call once on training set."""
        raw = self._detect_and_crop(images)
        all_patches = np.concatenate([r for r in raw if len(r) > 0], axis=0)
        self.pca = PCA(n_components=self.k_pca, whiten=False)
        self.pca.fit(all_patches)
        return self

    def transform(self, images: List[np.ndarray]) -> List[Features]:
        """Extract features (PCA must be fitted)."""
        if self.pca is None:
            raise RuntimeError("Call fit() before transform().")

        results: List[Features] = []
        for img in _maybe_tqdm(
            images,
            desc="Features (transform)",
            unit="img",
            enabled=self.progress and len(images) > 0,
        ):
            img = _to_grey(img)
            feats = kadir_brady(img, self.n_feat, self.s_min, self.s_max)
            if len(feats) == 0:
                results.append(Features(
                    np.zeros((0, 2)), np.zeros(0), np.zeros((0, self.k_pca))
                ))
                continue

            xs, ys, scales, _ = feats[:, 0], feats[:, 1], feats[:, 2], feats[:, 3]
            patches = _crop_patches(img, xs.astype(int), ys.astype(int),
                                    scales.astype(int), self.patch_sz)
            valid = [j for j, p in enumerate(patches) if p is not None]
            if not valid:
                results.append(Features(
                    np.zeros((0, 2)), np.zeros(0), np.zeros((0, self.k_pca))
                ))
                continue

            X_out = np.stack([feats[j, :2] for j in valid])    # (N,2) [x,y]
            S_out = np.array([np.log(max(feats[j, 2], 1))
                              for j in valid])                   # log-scale
            raw_patches = np.stack([patches[j] for j in valid]) # (N, patch²)
            A_out = self.pca.transform(raw_patches)             # (N, K)

            results.append(Features(X_out, S_out, A_out))
        return results

    def fit_transform(self, images: List[np.ndarray]) -> List[Features]:
        """
        Fit PCA and return features in one pass per image.

        The naive ``fit().transform()`` would run the Kadir-Brady detector twice
        per image; this implementation detects once (the dominant cost).
        """
        stash: List[Optional[Tuple[np.ndarray, List[int], List[Optional[np.ndarray]]]]] = []
        blocks: List[np.ndarray] = []

        for img in _maybe_tqdm(
            images,
            desc="Features (fit+PCA)",
            unit="img",
            enabled=self.progress and len(images) > 0,
        ):
            img = _to_grey(img)
            feats = kadir_brady(img, self.n_feat, self.s_min, self.s_max)
            if len(feats) == 0:
                stash.append(None)
                continue

            xs, ys, scales, _ = feats[:, 0], feats[:, 1], feats[:, 2], feats[:, 3]
            patches = _crop_patches(
                img, xs.astype(int), ys.astype(int), scales.astype(int), self.patch_sz
            )
            valid = [j for j, p in enumerate(patches) if p is not None]
            if not valid:
                stash.append(None)
                continue

            raw_stack = np.stack([patches[j] for j in valid])
            blocks.append(raw_stack)
            stash.append((feats, valid, patches))

        if not blocks:
            raise RuntimeError(
                "No salient regions in any training image; loosen --s_min/--s_max or check data."
            )

        all_patches = np.concatenate(blocks, axis=0)
        self.pca = PCA(n_components=self.k_pca, whiten=False)
        self.pca.fit(all_patches)

        results: List[Features] = []
        for inter in _maybe_tqdm(
            stash,
            desc="PCA → features",
            unit="img",
            enabled=self.progress and len(stash) > 0,
        ):
            if inter is None:
                results.append(
                    Features(
                        np.zeros((0, 2)), np.zeros(0), np.zeros((0, self.k_pca))
                    )
                )
                continue
            feats, valid, patches = inter
            X_out = np.stack([feats[j, :2] for j in valid])
            S_out = np.array([np.log(max(feats[j, 2], 1)) for j in valid])
            raw_patches = np.stack([patches[j] for j in valid])
            A_out = self.pca.transform(raw_patches)
            results.append(Features(X_out, S_out, A_out))
        return results

    # ------------------------------------------------------------------
    def _detect_and_crop(self, images: List[np.ndarray]) -> List[np.ndarray]:
        all_patches = []
        for img in _maybe_tqdm(
            images,
            desc="Features (fit detect)",
            unit="img",
            enabled=self.progress and len(images) > 0,
        ):
            img = _to_grey(img)
            feats = kadir_brady(img, self.n_feat, self.s_min, self.s_max)
            if len(feats) == 0:
                all_patches.append(np.zeros((0, self.patch_sz ** 2)))
                continue
            xs, ys, scales, _ = feats[:, 0], feats[:, 1], feats[:, 2], feats[:, 3]
            patches = _crop_patches(img, xs.astype(int), ys.astype(int),
                                    scales.astype(int), self.patch_sz)
            valid = [p for p in patches if p is not None]
            if valid:
                all_patches.append(np.stack(valid))
            else:
                all_patches.append(np.zeros((0, self.patch_sz ** 2)))
        return all_patches


# -----------------------------------------------------------------------
def _to_grey(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float64)
    if img.max() > 1.0:
        img /= 255.0
    if img.ndim == 3:
        img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    return img


def _crop_patches(
    img: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    scales: np.ndarray,
    patch_sz: int,
) -> List[Optional[np.ndarray]]:
    H, W = img.shape
    half = patch_sz // 2
    patches = []
    for x, y, s in zip(xs, ys, scales):
        r0, r1 = y - s, y + s
        c0, c1 = x - s, x + s
        if r0 < 0 or c0 < 0 or r1 >= H or c1 >= W or r1 <= r0 or c1 <= c0:
            patches.append(None)
            continue
        crop = img[r0:r1, c0:c1]
        patch_img = Image.fromarray((crop * 255).astype(np.uint8))
        patch_img = patch_img.resize((patch_sz, patch_sz), Image.BILINEAR)
        patch = np.array(patch_img, dtype=np.float64).ravel() / 255.0
        patches.append(patch)
    return patches


def load_images(paths: List[Path]) -> List[np.ndarray]:
    imgs = []
    for p in paths:
        try:
            img = np.array(Image.open(p).convert("L"), dtype=np.float64) / 255.0
            imgs.append(img)
        except Exception as e:
            print(f"Warning: could not load {p}: {e}")
    return imgs
