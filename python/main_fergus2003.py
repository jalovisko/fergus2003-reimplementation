#!/usr/bin/env python3
"""
main_fergus2003.py — Driver script for the Fergus et al. (2003) reimplementation.

Directory layout:
    data/
      motorbikes/    (*.jpg / *.png)
      airplanes/
      faces/
      cars_rear/
      background/

Usage:
    python main_fergus2003.py
    python main_fergus2003.py --category faces --parts 6 --iter 80
"""
from __future__ import annotations

import argparse
import logging
import pickle
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import roc_curve

from constellation_model import (
    ConstellationModel,
    ConstellationRecognizer,
    EMConstellationLearner,
)
from feature_extraction import FeatureExtractor, Features, load_images

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Fergus et al. 2003 — constellation model")
    parser.add_argument("--data_root",  default="data",           help="root data directory")
    parser.add_argument("--categories", nargs="+",
                        default=["motorbikes", "airplanes", "faces", "cars_rear"])
    parser.add_argument("--parts",      type=int, default=6,      help="number of parts P")
    parser.add_argument("--iter",       type=int, default=60,     help="EM iterations")
    parser.add_argument("--n_feat",     type=int, default=30,     help="features per image")
    parser.add_argument("--k_pca",      type=int, default=15,     help="PCA dimensions")
    parser.add_argument("--patch",      type=int, default=11,     help="patch size (pixels)")
    parser.add_argument("--s_min",      type=int, default=4,      help="min detector scale")
    parser.add_argument("--s_max",      type=int, default=40,     help="max detector scale")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--save_models", action="store_true",     help="pickle learnt models")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)

    data_root = Path(args.data_root)
    bg_dir    = data_root / "background"

    # ---- Background images -----------------------------------------------
    log.info("Loading background images from %s", bg_dir)
    bg_paths = sorted(list(bg_dir.glob("*.jpg")) + list(bg_dir.glob("*.png")))
    bg_imgs  = load_images(bg_paths)
    log.info("  %d background images loaded.", len(bg_imgs))

    results: Dict[str, dict] = {}

    for category in args.categories:
        log.info("\n" + "=" * 55)
        log.info("Category: %s", category)
        log.info("=" * 55)

        cat_dir = data_root / category
        paths   = sorted(list(cat_dir.glob("*.jpg")) + list(cat_dir.glob("*.png")))
        idx     = list(range(len(paths)))
        rng.shuffle(idx)
        n_train = len(idx) // 2
        train_paths = [paths[i] for i in idx[:n_train]]
        test_paths  = [paths[i] for i in idx[n_train:]]
        log.info("  train=%d  test=%d", len(train_paths), len(test_paths))

        # ---- Feature extraction ----------------------------------------
        log.info("  Extracting training features...")
        extractor = FeatureExtractor(
            n_feat=args.n_feat, patch_sz=args.patch,
            k_pca=args.k_pca, s_min=args.s_min, s_max=args.s_max,
        )
        train_imgs = load_images(train_paths)
        train_feats = extractor.fit_transform(train_imgs)

        bg_feats = extractor.transform(bg_imgs)

        # ---- EM learning -----------------------------------------------
        log.info("  Learning constellation model (P=%d, %d EM iters)...",
                 args.parts, args.iter)
        learner = EMConstellationLearner(P=args.parts, n_iter=args.iter, verbose=True)
        model   = learner.fit(train_feats, bg_feats)

        if args.save_models:
            out_path = Path(f"model_{category}.pkl")
            with open(out_path, "wb") as f:
                pickle.dump(model, f)
            log.info("  Saved model → %s", out_path)

        # ---- Recognition -----------------------------------------------
        log.info("  Recognizing test images...")
        recognizer = ConstellationRecognizer(model)

        test_imgs  = load_images(test_paths)
        test_feats = extractor.transform(test_imgs)
        R_fg = recognizer.predict(test_feats)

        n_bg_test = len(test_paths)
        bg_test_paths = random.sample(bg_paths, min(n_bg_test, len(bg_paths)))
        bg_test_imgs  = load_images(bg_test_paths)
        bg_test_feats = extractor.transform(bg_test_imgs)
        R_bg = recognizer.predict(bg_test_feats)

        # ---- Evaluation ------------------------------------------------
        scores = np.concatenate([R_fg, R_bg])
        labels = np.concatenate([np.ones(len(R_fg)), np.zeros(len(R_bg))])
        eer    = compute_eer(scores, labels)
        acc    = (1 - eer) * 100

        log.info("  ROC EER = %.1f%%  |  Accuracy ≈ %.1f%%", eer * 100, acc)

        results[category] = {
            "model": model, "extractor": extractor,
            "R_fg": R_fg, "R_bg": R_bg,
            "eer": eer, "acc": acc,
        }

        # ---- Shape model visualisation ---------------------------------
        plot_shape_model(model, category)
        plt.savefig(f"shape_model_{category}.png", dpi=150, bbox_inches="tight")
        plt.close()

    # ---- Summary ---------------------------------------------------------
    print("\n" + "=" * 45)
    print(f"{'Dataset':<20}  {'Accuracy (%)':>12}")
    print("-" * 35)
    for cat, r in results.items():
        print(f"{cat:<20}  {r['acc']:>12.1f}")
    print("=" * 45)

    plot_roc(results)
    plt.savefig("roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved roc_curves.png")


# ---------------------------------------------------------------------------
def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    fpr, tpr, _ = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr - fnr))
    return float((fpr[idx] + fnr[idx]) / 2)


def plot_shape_model(model: ConstellationModel, title: str):
    P      = model.P
    mu     = model.mu
    Sigma  = model.Sigma
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, P))
    for p in range(P):
        mx = mu[2 * p];     my = mu[2 * p + 1]
        sx = np.sqrt(Sigma[2*p,   2*p])
        sy = np.sqrt(Sigma[2*p+1, 2*p+1])
        theta = np.linspace(0, 2 * np.pi, 100)
        ax.plot(mx + sx * np.cos(theta), my + sy * np.sin(theta),
                color=colors[p], linewidth=2)
        ax.plot(mx, my, "o", color=colors[p], markersize=8)
        ax.text(mx + 0.01, my + 0.01, f"P{p+1}", color=colors[p], fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)
    ax.set_title(f"{title} — shape model (P={P})")
    ax.set_xlabel("x (normalised)")
    ax.set_ylabel("y (normalised)")


def plot_roc(results: dict):
    fig, ax = plt.subplots(figsize=(6, 5))
    from sklearn.metrics import roc_curve as _roc
    for cat, r in results.items():
        scores = np.concatenate([r["R_fg"], r["R_bg"]])
        labels = np.concatenate([np.ones(len(r["R_fg"])), np.zeros(len(r["R_bg"]))])
        fpr, tpr, _ = _roc(labels, scores)
        ax.plot(fpr, tpr, label=f"{cat} (acc={r['acc']:.1f}%)")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Fergus et al. (2003)")
    ax.legend()
    ax.grid(True, alpha=0.3)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
