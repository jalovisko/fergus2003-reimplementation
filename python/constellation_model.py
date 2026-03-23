"""
constellation_model.py
EM learning and Bayesian recognition for the
Fergus et al. (2003) constellation model.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
from feature_extraction import Features
from scipy.special import gammaln
from scipy.stats import multivariate_normal

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
@dataclass
class ConstellationModel:
    """Parameters of the learned constellation model."""
    P:   int = 6    # parts
    K:   int = 15   # PCA dims

    # Shape
    mu:    np.ndarray = field(default_factory=lambda: np.zeros(12))   # 2P
    Sigma: np.ndarray = field(default_factory=lambda: np.eye(12))     # 2P×2P

    # Per-part appearance: list of (mean [K], diag-cov [K])
    c: List[np.ndarray] = field(default_factory=list)
    V: List[np.ndarray] = field(default_factory=list)

    # Background appearance
    c_bg: np.ndarray = field(default_factory=lambda: np.zeros(15))
    V_bg: np.ndarray = field(default_factory=lambda: np.ones(15))

    # Per-part scale (log-scale)
    t: np.ndarray = field(default_factory=lambda: np.zeros(6))   # [P]
    U: np.ndarray = field(default_factory=lambda: np.ones(6))    # [P] variances

    # Feature statistics
    M: float = 10.0          # Poisson mean for background features

    # Occlusion table: 2^P entries
    occ_prob: np.ndarray = field(default_factory=lambda: np.full(64, 1/64))
    occ_states: np.ndarray = field(default_factory=lambda: _all_binary_states(6))

    def __post_init__(self):
        if not self.c:
            self.c = [np.zeros(self.K) for _ in range(self.P)]
            self.V = [np.ones(self.K) for _ in range(self.P)]


def _all_binary_states(P: int) -> np.ndarray:
    """Return (2^P, P) array of all binary vectors."""
    return np.array([[int(b) for b in format(i, f'0{P}b')] for i in range(2**P)])


# ---------------------------------------------------------------------------
class EMConstellationLearner:
    """
    EM algorithm for learning constellation model parameters.

    Parameters
    ----------
    P          : number of parts
    n_iter     : EM iterations
    n_hyps     : hypotheses sampled per image per E-step
    verbose    : print iteration log
    """

    def __init__(
        self,
        P: int       = 6,
        n_iter: int  = 60,
        n_hyps: int  = 200,
        verbose: bool = True,
    ):
        self.P       = P
        self.n_iter  = n_iter
        self.n_hyps  = n_hyps
        self.verbose = verbose

    # ------------------------------------------------------------------
    def fit(
        self,
        train_features: List[Features],
        bg_features:    List[Features],
    ) -> ConstellationModel:

        P = self.P
        # Infer K
        for f in train_features:
            if f.A.shape[0] > 0:
                K = f.A.shape[1]
                break
        else:
            raise ValueError("No valid training features found.")

        model = self._init_model(P, K, train_features, bg_features)

        for it in range(1, self.n_iter + 1):
            ss = self._e_step(model, train_features)
            model = self._m_step(model, ss, train_features)
            if self.verbose and it % 10 == 0:
                log.info("EM iter %d/%d  log-lik=%.2f", it, self.n_iter, ss["log_lik"])

        return model

    # ------------------------------------------------------------------
    def _init_model(
        self,
        P: int, K: int,
        train_features: List[Features],
        bg_features:    List[Features],
    ) -> ConstellationModel:

        model = ConstellationModel(P=P, K=K)
        model.occ_states = _all_binary_states(P)
        model.occ_prob   = np.full(2**P, 1.0 / 2**P)

        # Background appearance
        bg_A = np.concatenate([f.A for f in bg_features if f.A.shape[0] > 0], axis=0)
        if len(bg_A) > 0:
            model.c_bg = bg_A.mean(axis=0)
            model.V_bg = np.maximum(bg_A.var(axis=0), 1e-6)
        else:
            model.c_bg = np.zeros(K)
            model.V_bg = np.ones(K)

        # Foreground appearance: random seed from training patches
        all_A = np.concatenate([f.A for f in train_features if f.A.shape[0] > 0], axis=0)
        rng = np.random.default_rng(0)
        seed_idx = rng.choice(len(all_A), size=P, replace=False)
        model.c = [all_A[seed_idx[p]].copy() for p in range(P)]
        model.V = [np.maximum(all_A.var(axis=0) * 2, 1e-4) for _ in range(P)]

        # Shape
        model.mu    = rng.normal(0, 0.1, 2 * P)
        model.Sigma = np.eye(2 * P) * 0.3

        # Scale
        all_S = np.concatenate([f.S for f in train_features if f.S.shape[0] > 0])
        model.t = np.full(P, all_S.mean())
        model.U = np.full(P, max(all_S.var(), 1e-4))

        # Poisson mean
        avg_N = np.mean([f.X.shape[0] for f in train_features])
        model.M = max(avg_N - P, 1.0)

        return model

    # ------------------------------------------------------------------
    def _e_step(
        self,
        model: ConstellationModel,
        train_features: List[Features],
    ) -> dict:
        """Compute sufficient statistics."""
        P, K = model.P, model.K

        ss = {
            "A_sum":    np.zeros((K, P)),
            "A_sq":     np.zeros((K, P)),
            "weight":   np.zeros(P),
            "XS":       np.zeros(2 * P),
            "XX":       np.zeros((2 * P, 2 * P)),
            "t_sum":    np.zeros(P),
            "t_sq":     np.zeros(P),
            "occ_cnt":  np.zeros(2**P),
            "n_bg":     0.0,
            "log_lik":  0.0,
        }

        for feat in train_features:
            N = feat.X.shape[0]
            if N < 2:
                continue

            hyps, log_w = self._sample_hypotheses(feat, model)
            if len(hyps) == 0:
                continue

            # Normalise
            log_w -= log_w.max()
            w  = np.exp(log_w)
            w /= w.sum() + 1e-300
            ss["log_lik"] += float(np.log(np.exp(log_w).sum() + 1e-300))

            for hi, h in enumerate(hyps):
                d   = (h > 0).astype(int)
                whi = float(w[hi])

                # Occlusion
                occ_idx = int("".join(str(b) for b in d), 2)
                ss["occ_cnt"][occ_idx] += whi

                # Background feature count
                ss["n_bg"] += whi * (N - d.sum())

                # Reference part for scale-normalised shape
                ref_parts = np.where(d)[0]
                if len(ref_parts) == 0:
                    continue
                ref_p = ref_parts[0]
                s_ref = feat.S[h[ref_p]]

                loc = np.zeros(2 * P)
                for p in range(P):
                    if d[p]:
                        dx = feat.X[h[p], 0] - feat.X[h[ref_p], 0]
                        dy = feat.X[h[p], 1] - feat.X[h[ref_p], 1]
                        loc[2*p:2*p+2] = np.array([dx, dy]) * np.exp(-s_ref)
                    else:
                        loc[2*p:2*p+2] = model.mu[2*p:2*p+2]

                ss["XS"] += whi * loc
                ss["XX"] += whi * np.outer(loc, loc)

                for p in range(P):
                    if d[p]:
                        a = feat.A[h[p]]
                        ss["A_sum"][:, p]  += whi * a
                        ss["A_sq"][:, p]   += whi * a ** 2
                        ss["weight"][p]    += whi
                        ss["t_sum"][p]     += whi * feat.S[h[p]]
                        ss["t_sq"][p]      += whi * feat.S[h[p]] ** 2

        return ss

    # ------------------------------------------------------------------
    def _m_step(
        self,
        model: ConstellationModel,
        ss: dict,
        train_features: List[Features],
    ) -> ConstellationModel:
        P, K = model.P, model.K
        w_tot = ss["weight"] + 1e-10

        # Shape
        w_all = w_tot.sum()
        model.mu    = ss["XS"] / w_all
        model.Sigma = ss["XX"] / w_all - np.outer(model.mu, model.mu)
        model.Sigma = (model.Sigma + model.Sigma.T) / 2 + np.eye(2*P) * 1e-4

        # Appearance
        for p in range(P):
            model.c[p] = ss["A_sum"][:, p] / w_tot[p]
            v = ss["A_sq"][:, p] / w_tot[p] - model.c[p] ** 2
            model.V[p] = np.maximum(v, 1e-4)

        # Scale
        model.t = ss["t_sum"] / w_tot
        model.U = np.maximum(ss["t_sq"] / w_tot - model.t ** 2, 1e-4)

        # Poisson
        model.M = max(ss["n_bg"] / max(len(train_features), 1), 1.0)

        # Occlusion
        occ = ss["occ_cnt"] + 1e-6
        model.occ_prob = occ / occ.sum()

        return model

    # ------------------------------------------------------------------
    def _sample_hypotheses(
        self,
        feat: Features,
        model: ConstellationModel,
        n_mc: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Monte-Carlo hypothesis sampling with importance weighting."""
        if n_mc is None:
            n_mc = self.n_hyps

        P, K = model.P, model.K
        N = feat.X.shape[0]

        # Pre-compute log scores
        log_app    = np.zeros((N, P))
        log_app_bg = np.zeros(N)
        log_scl    = np.zeros((N, P))

        for j in range(N):
            log_app_bg[j] = _log_mvn_diag(feat.A[j], model.c_bg, model.V_bg)
            for p in range(P):
                log_app[j, p] = _log_mvn_diag(feat.A[j], model.c[p], model.V[p])
                log_scl[j, p] = _log_norm1d(feat.S[j], model.t[p], model.U[p])

        seen = set()
        hyps_list, lw_list = [], []

        for _ in range(n_mc * 12):
            if len(hyps_list) >= n_mc:
                break

            h = np.zeros(P, dtype=int)
            available = list(range(N))
            for p in range(P):
                if np.random.rand() < 0.15 or not available:
                    h[p] = 0
                else:
                    scores = (log_app[available, p] + log_scl[available, p])
                    scores -= scores.max()
                    probs = np.exp(scores)
                    probs /= probs.sum()
                    sel = np.random.choice(len(available), p=probs)
                    h[p] = available[sel]
                    available.pop(sel)

            key = tuple(h)
            if key in seen:
                continue
            seen.add(key)

            lw = _score_hypothesis(h, feat, model)
            hyps_list.append(h)
            lw_list.append(lw)

        if not hyps_list:
            return np.zeros((0, P), dtype=int), np.zeros(0)

        hyps = np.stack(hyps_list)
        lw   = np.array(lw_list)
        top  = min(50, len(lw))
        idx  = np.argsort(-lw)[:top]
        return hyps[idx], lw[idx]


# ---------------------------------------------------------------------------
class ConstellationRecognizer:
    """Compute log-likelihood ratio for a single image."""

    def __init__(self, model: ConstellationModel, n_hyps: int = 200):
        self.model  = model
        self.n_hyps = n_hyps
        self._learner = EMConstellationLearner(
            P=model.P, n_iter=0, n_hyps=n_hyps, verbose=False
        )

    def log_ratio(self, feat: Features) -> float:
        """Return log R = log p(fg) - log p(bg).  Positive => object present."""
        if feat.X.shape[0] == 0:
            return float("-inf")

        hyps, log_w = self._learner._sample_hypotheses(feat, self.model, self.n_hyps)
        if len(hyps) == 0:
            return float("-inf")

        # log p(X,S,A | theta) = log-sum-exp over hypotheses
        lse = log_w.max() + np.log(np.exp(log_w - log_w.max()).sum())

        # log p(X,S,A | bg): all features from background
        N = feat.X.shape[0]
        log_bg = sum(
            _log_mvn_diag(feat.A[j], self.model.c_bg, self.model.V_bg)
            for j in range(N)
        )
        log_bg += N * np.log(self.model.M + 1e-10) - self.model.M - float(gammaln(N + 1))

        return float(lse - log_bg)

    def predict(self, features: List[Features]) -> np.ndarray:
        return np.array([self.log_ratio(f) for f in features])


# ---------------------------------------------------------------------------
#  Utility functions
# ---------------------------------------------------------------------------
def _log_mvn_diag(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> float:
    K  = len(x)
    d  = x - mu
    return float(-0.5 * (K * np.log(2 * np.pi)
                         + np.sum(np.log(var + 1e-300))
                         + np.sum(d ** 2 / (var + 1e-300))))


def _log_norm1d(x: float, mu: float, sigma2: float) -> float:
    return float(-0.5 * (np.log(2 * np.pi * sigma2 + 1e-300)
                         + (x - mu) ** 2 / (sigma2 + 1e-300)))


def _log_mvn_full(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
    K = len(x)
    d = x - mu
    try:
        L  = np.linalg.cholesky(Sigma)
        lp = (-0.5 * (K * np.log(2 * np.pi)
                      + 2 * np.sum(np.log(np.diag(L)))
                      + np.sum(np.linalg.solve(L, d) ** 2)))
        return float(lp)
    except np.linalg.LinAlgError:
        return float("-inf")


def _score_hypothesis(
    h: np.ndarray,
    feat: Features,
    model: ConstellationModel,
) -> float:
    """Log-weight of a single hypothesis h (vector of length P, 0 = occluded)."""
    P = model.P
    N = feat.X.shape[0]
    d = (h > 0).astype(int)
    f = int(d.sum())
    n = N - f

    # Occlusion
    occ_idx = int("".join(str(b) for b in d), 2)
    log_occ = np.log(model.occ_prob[occ_idx] + 1e-300)

    # Poisson ratio
    log_poiss = (
        n * np.log(model.M + 1e-10) - model.M - float(gammaln(n + 1))
        - N * np.log(model.M + 1e-10) + model.M + float(gammaln(N + 1))
    )

    # Binomial book-keeping
    from math import comb
    log_binom = -np.log(comb(N, f) + 1e-300)

    # Appearance ratio
    log_app = 0.0
    for p in range(P):
        if d[p]:
            a = feat.A[h[p]]
            log_app += (_log_mvn_diag(a, model.c[p], model.V[p])
                        - _log_mvn_diag(a, model.c_bg, model.V_bg))

    # Shape (scale-normalised)
    ref_parts = np.where(d)[0]
    if len(ref_parts) == 0:
        log_shape = 0.0
    else:
        ref_p = ref_parts[0]
        s_ref = feat.S[h[ref_p]]
        loc = np.zeros(2 * P)
        for p in range(P):
            if d[p]:
                dx = feat.X[h[p], 0] - feat.X[h[ref_p], 0]
                dy = feat.X[h[p], 1] - feat.X[h[ref_p], 1]
                loc[2*p:2*p+2] = np.array([dx, dy]) * np.exp(-s_ref)
            else:
                loc[2*p:2*p+2] = model.mu[2*p:2*p+2]

        idx = []
        for p in range(P):
            if d[p]:
                idx += [2*p, 2*p+1]
        if len(idx) >= 2:
            mu_sub  = model.mu[idx]
            Sig_sub = model.Sigma[np.ix_(idx, idx)] + np.eye(len(idx)) * 1e-6
            log_shape = _log_mvn_full(loc[idx], mu_sub, Sig_sub) + f * np.log(1000)
        else:
            log_shape = 0.0

    # Scale ratio
    log_scale = 0.0
    for p in range(P):
        if d[p]:
            log_scale += _log_norm1d(feat.S[h[p]], model.t[p], model.U[p])
    log_scale += f * np.log(10)

    return log_occ + log_poiss + log_binom + log_app + log_shape + log_scale
