"""Uplift (heterogeneous treatment effect) modelling and evaluation.

An uplift model predicts tau(x) = E[Y | X=x, T=1] - E[Y | X=x, T=0]: how much
the treatment changes the outcome *for this user*. Ranking users by tau(x)
lets a business target only the people the treatment actually moves.

Learners implemented (Künzel et al. 2019 meta-learner taxonomy):
  - S-learner: one model with T as a feature
  - T-learner: separate outcome models per arm
  - X-learner: imputes individual effects, robust to unbalanced arms
  - Transformed-outcome (TO): regress Z = Y (T-p)/(p(1-p)), E[Z|X] = tau(x)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

_trapz = getattr(np, "trapezoid", None) or np.trapz


def _lgb_reg(**kw):
    import lightgbm as lgb

    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=500,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=-1)
    params.update(kw)
    return lgb.LGBMRegressor(**params)


def _lgb_clf(**kw):
    import lightgbm as lgb

    params = dict(n_estimators=300, learning_rate=0.05, num_leaves=31, min_child_samples=500,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, verbose=-1, n_jobs=-1)
    params.update(kw)
    return lgb.LGBMClassifier(**params)


def _prob(model, X):
    return model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else model.predict(X)


class SLearner:
    name = "S-learner"

    def fit(self, X, t, y):
        self.m = _lgb_clf().fit(np.column_stack([X, t]), y)
        return self

    def predict(self, X):
        ones, zeros = np.ones(len(X)), np.zeros(len(X))
        return _prob(self.m, np.column_stack([X, ones])) - _prob(self.m, np.column_stack([X, zeros]))


class TLearner:
    name = "T-learner"

    def fit(self, X, t, y):
        t = np.asarray(t).astype(bool)
        self.m1 = _lgb_clf().fit(X[t], y[t])
        self.m0 = _lgb_clf().fit(X[~t], y[~t])
        return self

    def predict(self, X):
        return _prob(self.m1, X) - _prob(self.m0, X)


class XLearner:
    """X-learner with a constant propensity (randomised experiment)."""

    name = "X-learner"

    def fit(self, X, t, y):
        t = np.asarray(t).astype(bool)
        y = np.asarray(y, dtype=float)
        self.p = t.mean()
        m1 = _lgb_clf().fit(X[t], y[t])
        m0 = _lgb_clf().fit(X[~t], y[~t])
        d1 = y[t] - _prob(m0, X[t])        # imputed effect for treated
        d0 = _prob(m1, X[~t]) - y[~t]      # imputed effect for control
        self.tau1 = _lgb_reg().fit(X[t], d1)
        self.tau0 = _lgb_reg().fit(X[~t], d0)
        return self

    def predict(self, X):
        # weight by propensity: the arm with more data gets less weight on its own imputation
        return self.p * self.tau0.predict(X) + (1 - self.p) * self.tau1.predict(X)


class TransformedOutcome:
    name = "Transformed outcome"

    def fit(self, X, t, y, e=None):
        """e: optional per-user propensity P(T=1|x); defaults to the overall treatment share."""
        t = np.asarray(t, dtype=float)
        p = t.mean() if e is None else np.asarray(e, dtype=float)
        self.p = float(np.mean(p))
        z = np.asarray(y, dtype=float) * (t - p) / (p * (1 - p))
        self.m = _lgb_reg(min_child_samples=2000).fit(X, z)
        return self

    def predict(self, X):
        return self.m.predict(X)


LEARNERS = {"S-learner": SLearner, "T-learner": TLearner, "X-learner": XLearner,
            "Transformed outcome": TransformedOutcome}


# ----------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------

def qini_curve(y, t, score, n_points: int = 101, weights=None):
    """Qini curve: incremental outcomes captured when targeting the top-k% by score.

    Q(k) = Y_t(k) - Y_c(k) * N_t(k) / N_c(k) (in units of the treated population).
    Optional ``weights`` (e.g. inverse-propensity weights t/e + (1-t)/(1-e)) make
    the curve valid when treatment probability varies with x. Returns (fractions, Q).
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t).astype(bool)
    w = np.ones(len(y)) if weights is None else np.asarray(weights, dtype=float)
    # Shuffle before a stable sort so tied scores (common with tree models) are broken at random
    # rather than by file order, which can be correlated with treatment.
    perm = np.random.default_rng(0).permutation(len(y))
    order = perm[np.argsort(-np.asarray(score)[perm], kind="stable")]
    y, t, w = y[order], t[order], w[order]
    cy_t = np.cumsum(w * y * t)
    cy_c = np.cumsum(w * y * ~t)
    cn_t = np.cumsum(w * t)
    cn_c = np.cumsum(w * ~t)
    idx = np.unique(np.clip((np.linspace(0, 1, n_points) * len(y)).astype(int) - 1, 0, len(y) - 1))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = cy_t[idx] - np.where(cn_c[idx] > 0, cy_c[idx] * cn_t[idx] / cn_c[idx], 0.0)
    frac = (idx + 1) / len(y)
    frac = np.concatenate([[0.0], frac])
    q = np.concatenate([[0.0], q])
    return frac, q


def qini_coefficient(y, t, score, weights=None) -> float:
    """Area between the model's Qini curve and the random-targeting line, normalised by Q(1).

    0 = no better than random; larger is better.
    """
    frac, q = qini_curve(y, t, score, weights=weights)
    if q[-1] == 0:
        return 0.0
    random_line = frac * q[-1]
    return float(_trapz(q - random_line, frac) / abs(q[-1]))


def share_captured(y, t, score, top_frac: float, weights=None) -> float:
    """Share of all incremental outcomes captured by targeting the top ``top_frac`` users."""
    frac, q = qini_curve(y, t, score, n_points=1001, weights=weights)
    return float(np.interp(top_frac, frac, q) / q[-1]) if q[-1] else float("nan")


def uplift_by_bins(y, t, score, n_bins: int = 10) -> pd.DataFrame:
    """Observed treatment effect within each predicted-uplift decile (1 = highest predicted)."""
    df = pd.DataFrame({"y": np.asarray(y, float), "t": np.asarray(t).astype(bool), "s": np.asarray(score)})
    jitter = np.random.default_rng(0).random(len(df)) * 1e-12  # random tie-breaking
    df["bin"] = pd.qcut((df["s"] + jitter).rank(method="first", ascending=False), n_bins,
                        labels=range(1, n_bins + 1))
    rows = []
    for b, g in df.groupby("bin", observed=True):
        yt, yc = g.loc[g.t, "y"], g.loc[~g.t, "y"]
        ate = yt.mean() - yc.mean()
        se = np.sqrt(yt.var(ddof=1) / max(len(yt), 1) + yc.var(ddof=1) / max(len(yc), 1))
        rows.append({"bin": int(b), "n": len(g), "pred_uplift": g.s.mean(), "rate_treated": yt.mean(),
                     "rate_control": yc.mean(), "observed_uplift": ate, "ci_low": ate - 1.96 * se,
                     "ci_high": ate + 1.96 * se})
    return pd.DataFrame(rows)


@dataclass
class UpliftEval:
    name: str
    qini: float
    top10: float
    top20: float
    top30: float
    fit_seconds: float

    def to_dict(self):
        return self.__dict__.copy()


def evaluate(name, y, t, score, fit_seconds=0.0, weights=None) -> UpliftEval:
    kw = dict(weights=weights)
    return UpliftEval(name, qini_coefficient(y, t, score, **kw), share_captured(y, t, score, 0.1, **kw),
                      share_captured(y, t, score, 0.2, **kw), share_captured(y, t, score, 0.3, **kw), fit_seconds)
