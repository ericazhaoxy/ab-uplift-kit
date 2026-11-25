"""Sequential testing: why 'peeking' breaks p-values, and mSPRT as the fix.

If you check a fixed-horizon p-value every day and stop at the first p < 0.05,
the false-positive rate is far above 5%. The mixture Sequential Probability
Ratio Test (mSPRT) gives *always-valid* p-values that can be monitored
continuously while keeping the false-positive rate at alpha.

Reference: Johari, Koomen, Pekelis, Walsh (2017/2022). Always Valid Inference:
Continuous Monitoring of A/B Tests. (The method behind Optimizely's Stats Engine.)
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def msprt_lambda(diff: np.ndarray, se: np.ndarray, tau: float) -> np.ndarray:
    """Gaussian-mixture likelihood ratio for H0: effect = 0, mixing N(0, tau^2).

    diff, se: estimated effect and its standard error at each look.
    """
    s2 = np.asarray(se, dtype=float) ** 2
    d = np.asarray(diff, dtype=float)
    t2 = tau**2
    log_lam = 0.5 * np.log(s2 / (s2 + t2)) + (t2 * d**2) / (2 * s2 * (s2 + t2))
    return np.exp(np.minimum(log_lam, 700))


def always_valid_pvalues(diff: np.ndarray, se: np.ndarray, tau: float) -> np.ndarray:
    """p_n = min(1, 1 / max_{k<=n} Lambda_k); non-increasing by construction."""
    lam = msprt_lambda(diff, se, tau)
    return np.minimum.accumulate(np.minimum(1.0, 1.0 / lam))


def naive_pvalues(diff: np.ndarray, se: np.ndarray) -> np.ndarray:
    z = np.asarray(diff) / np.asarray(se)
    return 2 * stats.norm.sf(np.abs(z))


def _cumulative_looks(y: np.ndarray, t: np.ndarray, n_looks: int):
    """Effect estimate and SE after each of n_looks equally sized batches (in given order)."""
    edges = np.linspace(0, len(y), n_looks + 1).astype(int)[1:]
    yt, yc = np.where(t, y, 0.0), np.where(~t, y, 0.0)
    cs = lambda a: np.cumsum(a)[edges - 1]
    n_t, n_c = cs(t.astype(float)), cs((~t).astype(float))
    s_t, s_c = cs(yt), cs(yc)
    q_t, q_c = cs(yt**2), cs(yc**2)
    n_t = np.maximum(n_t, 2)
    n_c = np.maximum(n_c, 2)
    m_t, m_c = s_t / n_t, s_c / n_c
    v_t = np.maximum(q_t / n_t - m_t**2, 1e-12) * n_t / (n_t - 1)
    v_c = np.maximum(q_c / n_c - m_c**2, 1e-12) * n_c / (n_c - 1)
    diff = m_t - m_c
    se = np.sqrt(v_t / n_t + v_c / n_c)
    return edges, diff, se


def sequential_path(y: np.ndarray, t: np.ndarray, n_looks: int = 50, tau: float | None = None, seed: int = 0) -> dict:
    """Simulate monitoring a real experiment over time (rows shuffled into arrival order)."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(y))
    y = np.asarray(y, dtype=float)[order]
    t = np.asarray(t).astype(bool)[order]
    n, diff, se = _cumulative_looks(y, t, n_looks)
    if tau is None:
        tau = default_tau(y, t)
    p_naive = naive_pvalues(diff, se)
    p_msprt = always_valid_pvalues(diff, se, tau)
    return {"n": n, "diff": diff, "se": se, "p_naive": p_naive, "p_msprt": p_msprt, "tau": tau}


def default_tau(y: np.ndarray, t: np.ndarray) -> float:
    """Heuristic mixing scale: ~10% of the baseline mean (a 'plausible' effect size).

    tau only affects power (how fast true effects are detected), never validity.
    """
    base = float(np.mean(np.asarray(y)[~np.asarray(t).astype(bool)]))
    return max(abs(base) * 0.1, 1e-6)


def simulate_peeking(y: np.ndarray, n_sims: int = 500, n_per_sim: int = 100_000, n_looks: int = 20,
                     treat_share: float = 0.5, alpha: float = 0.05, tau: float | None = None,
                     effect: float = 0.0, seed: int = 0) -> dict:
    """A/A (effect=0) or A/B simulation drawn from a real outcome distribution.

    Each simulation samples users, assigns arms at random, and checks the test at
    n_looks interim points. Returns the rejection rate of
      - fixed: one test at the planned end (the correct fixed-horizon test)
      - naive_peeking: reject if ANY interim fixed-horizon p < alpha
      - msprt: reject if the always-valid p ever drops below alpha
    For effect=0 these are false-positive rates; for effect>0 they are power.
    ``effect`` is added (as an absolute shift) to treated users' outcomes.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=float)
    if tau is None:
        tau = max(abs(y.mean()) * 0.1, 1e-6)
    rej = {"fixed": 0, "naive_peeking": 0, "msprt": 0}
    first_look_naive = []
    for _ in range(n_sims):
        idx = rng.integers(0, len(y), n_per_sim)
        yy = y[idx].copy()
        tt = rng.random(n_per_sim) < treat_share
        if effect:
            # Bernoulli-preserving shift for binary data, additive otherwise
            if set(np.unique(y)) <= {0.0, 1.0}:
                flip = (~yy.astype(bool)) & tt & (rng.random(n_per_sim) < effect / max(1 - y.mean(), 1e-9))
                yy[flip] = 1.0
            else:
                yy[tt] += effect
        _, diff, se = _cumulative_looks(yy, tt, n_looks)
        pn = naive_pvalues(diff, se)
        pm = always_valid_pvalues(diff, se, tau)
        rej["fixed"] += pn[-1] < alpha
        hit = np.flatnonzero(pn < alpha)
        if hit.size:
            rej["naive_peeking"] += 1
            first_look_naive.append(int(hit[0]) + 1)
        rej["msprt"] += pm[-1] < alpha
    out = {k: v / n_sims for k, v in rej.items()}
    out.update(n_sims=n_sims, n_per_sim=n_per_sim, n_looks=n_looks, tau=tau, effect=effect)
    return out
