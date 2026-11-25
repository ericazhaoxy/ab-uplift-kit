"""Power analysis and sample-size planning, including the effect of variance reduction."""
from __future__ import annotations

import numpy as np
from scipy import stats


def sample_size_per_arm(sigma: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8,
                        ratio: float = 1.0, variance_reduction: float = 0.0) -> tuple[int, int]:
    """Sample sizes (control, treatment) to detect an absolute effect ``mde_abs``.

    ratio = n_treatment / n_control. variance_reduction in [0, 1) shrinks the
    outcome variance, e.g. 0.3 from CUPED means 30% fewer users needed.
    """
    var = sigma**2 * (1 - variance_reduction)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    n_c = z**2 * var * (1 + 1 / ratio) / mde_abs**2
    return int(np.ceil(n_c)), int(np.ceil(n_c * ratio))


def sample_size_proportion(p_base: float, mde_rel: float, **kw) -> tuple[int, int]:
    """Sample size for a conversion-rate metric with relative MDE (e.g. 0.05 = +5%)."""
    sigma = np.sqrt(p_base * (1 - p_base))
    return sample_size_per_arm(sigma, p_base * mde_rel, **kw)


def mde(sigma: float, n_control: int, n_treatment: int, alpha: float = 0.05, power: float = 0.8,
        variance_reduction: float = 0.0) -> float:
    """Minimum detectable absolute effect for given sample sizes."""
    var = sigma**2 * (1 - variance_reduction)
    z = stats.norm.ppf(1 - alpha / 2) + stats.norm.ppf(power)
    return float(z * np.sqrt(var / n_control + var / n_treatment))


def power_at(sigma: float, effect_abs: float, n_control: int, n_treatment: int, alpha: float = 0.05,
             variance_reduction: float = 0.0) -> float:
    """Power of a two-sided z-test for a true absolute effect."""
    se = sigma * np.sqrt((1 - variance_reduction) * (1 / n_control + 1 / n_treatment))
    zc = stats.norm.ppf(1 - alpha / 2)
    shift = effect_abs / se
    return float(stats.norm.sf(zc - shift) + stats.norm.cdf(-zc - shift))
