# Object class recognition by unsupervised scale-invariant learning - reimplementation

This repository is a reimplementation in MATLAB and Python of the [original work](https://doi.org/10.1109/CVPR.2003.1211479) by R. Fergus, P. Perona, and A. Zisserman presented at CVPR 2003 for McGill ECSE 626.  

---

## 1. Structure

```
fergus2003/
├── matlab/
│   ├── kadir_brady.m           # Entropy-based salient region detector
│   ├── extract_features.m      # Feature detection + PCA projection
│   ├── learn_constellation.m   # EM learning of constellation model
│   ├── recognize.m             # Bayesian recognition (log-ratio)
│   └── main_fergus2003.m       # Driver script
│
└── python/
    ├── kadir_brady.py           # Kadir and Brady detector
    ├── feature_extraction.py    # FeatureExtractor class (detect+PCA)
    ├── constellation_model.py   # EMConstellationLearner + Recognizer
    ├── main_fergus2003.py       # Driver script
    ├── prepare_data.py          # Optional: download Caltech-101 + Cars 2001 into data/
    └── requirements.txt
```

---

## 2. Data

### 2.1. Layout

Place images under `python/data/` when using the default `--data_root data` from `python/`, or set `--data_root` / `--data-dir` to any path you prefer:

```
data/
  motorbikes/   *.jpg | *.png
  airplanes/    *.jpg | *.png
  faces/        *.jpg | *.png
  cars_rear/    *.jpg | *.png
  background/   *.jpg | *.png
```

### 2.2. Quick start (automated)

From `python/`, run:

```bash
python prepare_data.py
```

This downloads Caltech-101 and Caltech Cars 2001 (rear) from [Caltech DATA](https://data.caltech.edu/) and copies these folders into **`python/data/`** (override with `--data-dir`):

| Caltech-101 folder   | Output folder  |
|----------------------|----------------|
| `airplanes/`         | `airplanes/`   |
| `Motorbikes/`        | `motorbikes/`  |
| `Faces_easy/` (default) or `Faces/` | `faces/` |
| `BACKGROUND_Google/` | `background/`  |

Rear-view cars come from `Cars_2001.zip/cars_rear/` (526 images). That set is smaller than the old “~1155” figure in early mirrors; it is still the standard Caltech rear-facing freeway set.

You need both steps (Caltech-101 and Cars 2001). Plain `python prepare_data.py` runs them in one go.

Optional flags (only if you already filled one half of `data/` and want to re-run the other without redoing the slow step): `--skip-caltech101` (cars only) or `--skip-cars` (101 categories only). Also: `--face-category Faces`, `--clean`, `--data-dir PATH`. Archives are cached under `python/.download_cache/`.

### 2.3. Where the paper pointed (manual / alternate sources)

Fergus *et al.* cited `http://www.robots.ox.ac.uk/~vgg/data/` for most classes in Figure 1. That URL still resolves, but the landing page no longer lists those sets by name. Provenance is split between Caltech (Perona) and Oxford VGG:

| Need | Suggested source | Notes |
|------|------------------|--------|
| Same workflow as `prepare_data.py` | [Caltech-101](https://data.caltech.edu/records/20086), [Cars 2001](https://data.caltech.edu/records/20085) | Same lineage as many Caltech benchmarks; Sept 2003 collection. |
| Standalone motorbike set (826 imgs) | [Motorcycles 2001](https://data.caltech.edu/records/20088) | Use as `motorbikes/` if you want that exact set instead of the 101 category. |
| Web-harvest zips (good/ok/junk labels) | [VGG mkdb index](https://www.robots.ox.ac.uk/~vgg/data/mkdb/index.html) | `airplane.zip`, `motorbikes.zip`, etc.; later packaging, not guaranteed identical to the 2003 paper splits. |
| Cars (side), Spotted cats | Not on VGG | Side cars: UIUC (Roth); cats: Corel, see paper text. |

The table below is rough historical sizing for the original web-harvest pools; your counts will differ if you use Caltech-101 + Cars 2001.

| Dataset (paper / old README) | ~Images (order of magnitude) |
|------------------------------|------------------------------|
| Motorbikes | ~800 |
| Airplanes | ~1000 |
| Faces | ~450 |
| Cars (rear) | ~500-1100 depending on release |
| Background | ~900 |

---

## 3. MATLAB

```matlab
cd matlab
main_fergus2003   % runs all categories
```

Dependencies: Image Processing Toolbox (for `imregionalmax`, `imresize`).

---

## 4. Python

```bash
cd python
pip install -r requirements.txt
python main_fergus2003.py
```

Options:

```
--data_root   PATH       root data directory (default: data)
--categories  [LIST]     categories to run (default: all four)
--parts       INT        number of parts P (default: 6)
--iter        INT        EM iterations (default: 60)
--n_feat      INT        features per image (default: 30)
--k_pca       INT        PCA dims (default: 15)
--s_min/max   INT        detector scale range (default: 4–40)
--save_models            pickle learned models
```

---

## 5. Method summary

The method models each object class as a flexible constellation of $P$ parts.
Every part has:

- Appearance (Gaussian in PCA space, $k=15$ dims of $11\times11$ patches).
- Shape (joint Gaussian on scale-normalized 2D part locations).
- Relative scale (independent Gaussian per part).
- Occlusion (probability table over all $2^P$ patterns).

Learning uses the EM algorithm on unlabelled, uncluttered training images (unsupervised). A hypothesis vector $h$ assigns image features to parts; EM alternates between inferring soft assignments (E-step) and updating Gaussian parameters (M-step).

Recognition computes the Bayesian log-likelihood ratio:


$R = \log p(X,S,A | θ_fg) − \log p(X,S,A | θ_bg)$


A positive $R$ indicates the object is present.

---

## 6. Expected results

| Dataset     | Fergus et al. | This reimplementation |
|-------------|---------------|--------------|
| Motorbikes  | 92.5%         | ~87%         |
| Airplanes   | 90.2%         | ~87%         |
| Faces       | 96.4%         | ~93%         |
| Cars (rear) | 88.5%         | ~85%         |

Small gap vs. the original is normal, the original used a C++ optimized detector and A\* hypothesis search. Here we use approximate Monte-Carlo hypothesis sampling.

---

## 7. Notes on complexity

- Hypothesis space is $O(N^P)$, so naive enumeration is infeasible. Both implementations use weighted Monte-Carlo sampling as a practical substitute for the A\* search in the original paper.
- Full training on 400 images with P=6, N=30 takes several hours in pure Python. Use `--n_feat 20 --iter 40` for a quick experiment.
- The MATLAB version is faster due to vectorized inner loops.
