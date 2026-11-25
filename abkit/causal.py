"""Randomisation diagnostics and doubly-robust effect estimation.

Why this exists
---------------
A passing SRM check only says the *overall* split is right. If an experiment
pools several tests that used different treatment ratios (or assignment leaked
into features), treatment becomes correlated with user covariates. Then the
plain difference in means is biased, and CUPED-style adjustment *moves the
point estimate* instead of only shrinking the CI. The fix is to estimate the
propensity e(x) = P(T=1 | x) and use a doubly-robust estimator.

Reference: Chernozhukov et al. (2018), Double/debiased machine learning;
Robins, Rotnitzky & Zhao (1994), AIPW.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def _lgb_clf(**kw):
    import lightgbm as lgb

    params = dict(n_estimators=200, learning_rate=0.1, num_leaves=63, min_child_samples=500, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
    params.update(kw)
    return lgb.LGBMClassifier(**params)


def balance_table(X: np.ndarray, t: np.ndarray, names=None) -> pd.DataFrame:
    """Standardised mean difference per covariate. |SMD| > 0.1 is the usual red flag,
    but at millions of rows even 0.01 is far beyond chance (chance SMD ~ 1/sqrt(n))."""
    X = np.asarray(X, dtype=np.float64)
    t = np.asarray(t).astype(bool)
    names = names or [f"x{i}" for i in range(X.shape[1])]
    m1, m0 = X[t].mean(0), X[~t].mean(0)
    sd = np.sqrt((X[t].var(0) + X[~t].var(0)) / 2)
    smd = (m1 - m0) / np.where(sd > 0, sd, 1)
    se_smd = np.sqrt(1 / t.sum() + 1 / (~t).sum())
    return pd.DataFrame({"feature": names, "mean_control": m0, "mean_treatment": m1, "smd": smd,
                         "z": smd / se_smd}).reset_index(drop=True)


def cross_fit_nuisances(X, t, ys: dict, n_splits: int = 3, seed: int = 0, clip: float = 0.01):
    """Out-of-fold propensity e(x) and arm-specific outcome models mu1(x), mu0(x) for each outcome."""
    X = np.asarray(X)
    t = np.asarray(t).astype(bool)
    n = len(t)
    e = np.zeros(n)
    mu = {k: (np.zeros(n), np.zeros(n)) for k in ys}
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X):
        pm = _lgb_clf().fit(X[tr], t[tr])
        e[te] = pm.predict_proba(X[te])[:, 1]
        tr1, tr0 = tr[t[tr]], tr[~t[tr]]
        for k, y in ys.items():
            m1 = _lgb_clf().fit(X[tr1], y[tr1])
            m0 = _lgb_clf().fit(X[tr0], y[tr0])
            mu[k][1][te] = m1.predict_proba(X[te])[:, 1]
            mu[k][0][te] = m0.predict_proba(X[te])[:, 1]
    return np.clip(e, clip, 1 - clip), mu


def propensity_diagnostics(t, e) -> dict:
    t = np.asarray(t).astype(bool)
    q = np.quantile(e, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {"auc": float(roc_auc_score(t, e)), "e_q01": q[0], "e_q05": q[1], "e_median": q[2], "e_q95": q[3],
            "e_q99": q[4], "e_min": float(e.min()), "e_max": float(e.max())}


def aipw(y, t, e, mu1, mu0, alpha: float = 0.05) -> dict:
    """Augmented IPW (doubly robust) ATE with influence-function SEs and delta-method relative lift."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t, dtype=float)
    phi1 = mu1 + t * (y - mu1) / e
    phi0 = mu0 + (1 - t) * (y - mu0) / (1 - e)
    n = len(y)
    th1, th0 = phi1.mean(), phi0.mean()
    ate = th1 - th0
    cov = np.cov(phi1, phi0) / n
    se = float(np.sqrt(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]))
    z = stats.norm.ppf(1 - alpha / 2)
    g = np.array([1 / th0, -th1 / th0**2])
    se_rel = float(np.sqrt(g @ cov @ g))
    rel = th1 / th0 - 1
    return {"method": "aipw", "mean_control": float(th0), "mean_treatment": float(th1), "abs_lift": float(ate),
            "abs_ci_low": float(ate - z * se), "abs_ci_high": float(ate + z * se), "se": se,
            "rel_lift": float(rel), "rel_ci_low": float(rel - z * se_rel), "rel_ci_high": float(rel + z * se_rel),
            "p_value": float(2 * stats.norm.sf(abs(ate / se))), "n": int(n)}


def ipw(y, t, e, alpha: float = 0.05) -> dict:
    """Hajek (normalised) IPW estimate — unbiased under conditional randomisation, but noisy."""
    y = np.asarray(y, dtype=float)
    t = np.asarray(t).astype(bool)
    w1, w0 = t / e, (~t) / (1 - e)
    th1, th0 = (w1 * y).sum() / w1.sum(), (w0 * y).sum() / w0.sum()
    # linearised influence functions for SE
    phi1 = w1 * (y - th1) / w1.mean()
    phi0 = w0 * (y - th0) / w0.mean()
    se = float(np.sqrt(np.var(phi1 - phi0) / len(y)))
    ate = th1 - th0
    z = stats.norm.ppf(1 - alpha / 2)
    return {"method": "ipw", "mean_control": float(th0), "mean_treatment": float(th1), "abs_lift": float(ate),
            "abs_ci_low": float(ate - z * se), "abs_ci_high": float(ate + z * se), "se": se,
            "rel_lift": float(th1 / th0 - 1), "p_value": float(2 * stats.norm.sf(abs(ate / se)))}
