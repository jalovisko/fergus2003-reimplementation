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
    └── requirements.txt
```

---

## 2. Data

Download from [Caltech Vision Group](http://www.robots.ox.ac.uk/~vgg/data/):

| Dataset         | ~Images |
|-----------------|---------|
| Motorbikes      | 826     |
| Airplanes       | 1074    |
| Faces           | 450     |
| Cars (rear)     | 1155    |
| Background      | 900     |

Place them under `data/` as:

```
data/
  motorbikes/   *.jpg
  airplanes/    *.jpg
  faces/        *.jpg
  cars_rear/    *.jpg
  background/   *.jpg
```

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

| Dataset     | Fergus et al. | This reimplimentation |
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
