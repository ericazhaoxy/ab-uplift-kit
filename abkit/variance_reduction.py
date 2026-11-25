"""Variance reduction: CUPED, multi-covariate regression adjustment, and CUPAC.

All methods use only *pre-treatment* covariates, so they cannot bias the
treatment-effect estimate; they only remove noise that the covariates explain.

References
----------
- Deng, Xu, Kohavi, Walker (2013). Improving the Sensitivity of Online
  Controlled Experiments by Utilizing Pre-Experiment Data. (CUPED)
- Lin (2013). Agnostic notes on regression adjustments to experimental data.
- Tang et al. / DoorDash (2020). CUPAC: control using predictions as covariates.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import KFold

from .stats import TestResult, summary_from_moments, two_sample_test


@dataclass
class VarianceReductionResult:
    raw: TestResult
    adjusted: TestResult
    method: str

    @property
    def variance_reduction(self) -> float:
        """Share of the variance of the lift estimate removed (0..1)."""
        return 1.0 - (self.adjusted.se / self.raw.se) ** 2

    @property
    def sample_size_saving(self) -> float:
        """Required sample size scales with variance, so savings == variance reduction."""
        return self.variance_reduction

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "variance_reduction": self.variance_reduction,
            "sample_size_saving": self.sample_size_saving,
            "raw": self.raw.to_dict(),
            "adjusted": self.adjusted.to_dict(),
        }


def cuped(y: np.ndarray, t: np.ndarray, x: np.ndarray, metric: str = "metric", alpha: float = 0.05) -> VarianceReductionResult:
    """Classic CUPED with a single pre-period covariate x.

    theta = Cov(y, x) / Var(x) pooled over both arms; y_adj = y - theta (x - mean(x)).
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t).astype(bool)
    x = np.asarray(x, dtype=float)
    xc = x - x.mean()
    var_x = xc.var()
    theta = 0.0 if var_x == 0 else float(np.mean((y - y.mean()) * xc) / var_x)
    y_adj = y - theta * xc
    raw = two_sample_test(y[~t], y[t], metric=metric, alpha=alpha)
    adj = two_sample_test(y_adj[~t], y_adj[t], metric=metric, alpha=alpha)
    adj.method = "cuped"
    return VarianceReductionResult(raw=raw, adjusted=adj, method="cuped")


def regression_adjustment(y: np.ndarray, t: np.ndarray, X: np.ndarray, metric: str = "metric", alpha: float = 0.05) -> VarianceReductionResult:
    """Lin (2013) fully-interacted regression adjustment with many covariates.

    Fits a separate OLS per arm on covariates centred at the pooled mean, so
    each arm's intercept is its covariate-adjusted mean. Robust to unequal
    allocation (e.g. Criteo's 85/15 split) and never less efficient than the
    unadjusted estimator asymptotically.
    """
    y = np.asarray(y, dtype=float)
    t = np.asarray(t).astype(bool)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X[:, None]
    x_mean = X.mean(axis=0, dtype=np.float64)

    # Accumulate the normal equations (A'A, A'y, y'y) per arm in chunks, so
    # memory stays flat even for tens of millions of rows.
    k = X.shape[1] + 1
    AtA = {False: np.zeros((k, k)), True: np.zeros((k, k))}
    Aty = {False: np.zeros(k), True: np.zeros(k)}
    yty = {False: 0.0, True: 0.0}
    chunk = 1_000_000
    for s in range(0, len(y), chunk):
        Xc = X[s:s + chunk].astype(np.float64) - x_mean
        A_all = np.column_stack([np.ones(len(Xc)), Xc])
        yc, tc = y[s:s + chunk], t[s:s + chunk]
        for arm in (False, True):
            m = tc == arm
            A, ya = A_all[m], yc[m]
            AtA[arm] += A.T @ A
            Aty[arm] += A.T @ ya
            yty[arm] += ya @ ya

    means, var_means, ns = [], [], []
    for arm in (False, True):
        beta = np.linalg.solve(AtA[arm], Aty[arm])
        rss = max(yty[arm] - 2 * beta @ Aty[arm] + beta @ AtA[arm] @ beta, 0.0)
        n = int(t.sum()) if arm else int((~t).sum())
        dof = max(n - k, 1)
        means.append(beta[0])
        var_means.append(rss / dof / n)
        ns.append(n)

    raw = two_sample_test(y[~t], y[t], metric=metric, alpha=alpha)
    adj = summary_from_moments(metric, means[0], means[1], var_means[0], var_means[1], ns[0], ns[1], alpha, method="regression_adjustment")
    return VarianceReductionResult(raw=raw, adjusted=adj, method="regression_adjustment")


def cross_fit_predictions(X: np.ndarray, y: np.ndarray, model_factory, n_splits: int = 3, seed: int = 0) -> np.ndarray:
    """Out-of-fold predictions of y from pre-treatment X (treatment NOT used).

    Cross-fitting ensures each row's prediction comes from a model that never
    saw that row, which keeps the CUPAC estimator unbiased and its variance honest.
    """
    X = np.asarray(X)
    y = np.asarray(y, dtype=float)
    pred = np.zeros(len(y), dtype=float)
    for train_idx, test_idx in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X):
        model = model_factory()
        model.fit(X[train_idx], y[train_idx])
        if hasattr(model, "predict_proba") and len(np.unique(y)) == 2:
            pred[test_idx] = model.predict_proba(X[test_idx])[:, 1]
        else:
            pred[test_idx] = model.predict(X[test_idx])
    return pred


def default_model_factory(binary: bool = True):
    import lightgbm as lgb

    params = dict(n_estimators=200, learning_rate=0.1, num_leaves=63, min_child_samples=200, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=-1)
    if binary:
        return lambda: lgb.LGBMClassifier(**params)
    return lambda: lgb.LGBMRegressor(**params)


def cupac(y: np.ndarray, t: np.ndarray, X: np.ndarray, metric: str = "metric", alpha: float = 0.05,
          model_factory=None, n_splits: int = 3, seed: int = 0, return_pred: bool = False):
    """CUPAC: use an ML prediction of y (from pre-treatment X) as the CUPED covariate.

    Captures non-linear relationships that linear regression adjustment misses.
    """
    y = np.asarray(y, dtype=float)
    if model_factory is None:
        model_factory = default_model_factory(binary=set(np.unique(y)) <= {0.0, 1.0})
    pred = cross_fit_predictions(X, y, model_factory, n_splits=n_splits, seed=seed)
    res = cuped(y, t, pred, metric=metric, alpha=alpha)
    res.method = "cupac"
    res.adjusted.method = "cupac"
    return (res, pred) if return_pred else res
