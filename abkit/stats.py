"""Core two-sample inference for A/B tests.

All functions work on raw outcome arrays (binary or continuous) and return
plain dataclasses so they are easy to serialise into reports.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy import stats


@dataclass
class TestResult:
    metric: str
    mean_control: float
    mean_treatment: float
    abs_lift: float          # treatment - control
    abs_ci_low: float
    abs_ci_high: float
    rel_lift: float          # (treatment - control) / control
    rel_ci_low: float
    rel_ci_high: float
    se: float                # standard error of abs_lift
    z: float
    p_value: float
    n_control: int
    n_treatment: int
    method: str = "welch"

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05


def _rel_lift_ci(m_c, m_t, var_c, var_t, alpha):
    """Delta-method CI for the relative lift m_t / m_c - 1.

    var_c, var_t are variances of the *means* (i.e. s^2 / n).
    """
    if m_c == 0:
        return np.nan, np.nan, np.nan
    rel = m_t / m_c - 1.0
    # Var(m_t/m_c) ≈ var_t/m_c^2 + m_t^2 var_c / m_c^4  (independent groups)
    var_ratio = var_t / m_c**2 + (m_t**2) * var_c / m_c**4
    z = stats.norm.ppf(1 - alpha / 2)
    half = z * np.sqrt(var_ratio)
    return rel, rel - half, rel + half


def two_sample_test(
    y_control: np.ndarray,
    y_treatment: np.ndarray,
    metric: str = "metric",
    alpha: float = 0.05,
) -> TestResult:
    """Welch / two-proportion z-test (identical for large n) with delta-method relative CI."""
    y_c = np.asarray(y_control, dtype=float)
    y_t = np.asarray(y_treatment, dtype=float)
    n_c, n_t = len(y_c), len(y_t)
    m_c, m_t = y_c.mean(), y_t.mean()
    var_c = y_c.var(ddof=1) / n_c
    var_t = y_t.var(ddof=1) / n_t
    return _assemble(metric, m_c, m_t, var_c, var_t, n_c, n_t, alpha, "welch")


def _assemble(metric, m_c, m_t, var_c, var_t, n_c, n_t, alpha, method) -> TestResult:
    diff = m_t - m_c
    se = float(np.sqrt(var_c + var_t))
    z = diff / se if se > 0 else 0.0
    p = 2 * stats.norm.sf(abs(z))
    zc = stats.norm.ppf(1 - alpha / 2)
    rel, rl, rh = _rel_lift_ci(m_c, m_t, var_c, var_t, alpha)
    return TestResult(
        metric=metric,
        mean_control=float(m_c),
        mean_treatment=float(m_t),
        abs_lift=float(diff),
        abs_ci_low=float(diff - zc * se),
        abs_ci_high=float(diff + zc * se),
        rel_lift=float(rel),
        rel_ci_low=float(rl),
        rel_ci_high=float(rh),
        se=se,
        z=float(z),
        p_value=float(p),
        n_control=int(n_c),
        n_treatment=int(n_t),
        method=method,
    )


def summary_from_moments(metric, m_c, m_t, var_mean_c, var_mean_t, n_c, n_t, alpha=0.05, method="adjusted"):
    """Build a TestResult from already-computed means and variances of the means."""
    return _assemble(metric, m_c, m_t, var_mean_c, var_mean_t, n_c, n_t, alpha, method)
