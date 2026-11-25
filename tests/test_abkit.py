import numpy as np
import pytest
from scipy import stats

from abkit import (cupac, cuped, regression_adjustment, sample_size_per_arm, simulate_peeking, srm_check,
                   two_sample_test, power_at, qini_coefficient, LEARNERS)
from abkit.data import FEATURES, make_synthetic


def test_srm_balanced_and_broken():
    assert not srm_check([50_120, 49_880], [0.5, 0.5]).srm_detected
    assert srm_check([51_000, 49_000], [0.5, 0.5]).srm_detected
    assert not srm_check([850_300, 149_700], [0.85, 0.15]).srm_detected


def test_two_sample_matches_welch():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 5000), rng.normal(0.05, 1.3, 7000)
    r = two_sample_test(a, b)
    p_scipy = stats.ttest_ind(b, a, equal_var=False).pvalue
    assert r.p_value == pytest.approx(p_scipy, rel=0.02)
    assert r.abs_ci_low < r.abs_lift < r.abs_ci_high


def _correlated(n, effect, rng):
    x = rng.normal(size=n)
    t = rng.random(n) < 0.5
    y = 2 * x + rng.normal(size=n) + effect * t
    return y, t, x


def test_cuped_reduces_variance_and_is_unbiased():
    rng = np.random.default_rng(2)
    ests = []
    for _ in range(200):
        y, t, x = _correlated(4000, 0.1, rng)
        r = cuped(y, t, x)
        ests.append(r.adjusted.abs_lift)
    assert np.mean(ests) == pytest.approx(0.1, abs=0.01)
    assert r.variance_reduction > 0.7  # corr^2 = 0.8


def test_cuped_aa_false_positive_rate():
    rng = np.random.default_rng(3)
    rej = [cuped(*_correlated(2000, 0.0, rng)).adjusted.p_value < 0.05 for _ in range(600)]
    assert 0.03 < np.mean(rej) < 0.075


def test_regression_adjustment_matches_interacted_ols():
    rng = np.random.default_rng(4)
    n = 6000
    X = rng.normal(size=(n, 3))
    t = rng.random(n) < 0.3
    y = X @ [1.0, -0.5, 0.2] + t * (0.3 + 0.5 * X[:, 0]) + rng.normal(size=n)
    r = regression_adjustment(y, t, X)
    Xc = X - X.mean(0)
    A = np.column_stack([np.ones(n), t, Xc, Xc * t[:, None]])
    beta = np.linalg.lstsq(A, y, rcond=None)[0]
    assert r.adjusted.abs_lift == pytest.approx(beta[1], abs=1e-8)
    assert r.adjusted.se < r.raw.se


def test_power_formula_matches_simulation():
    sigma, eff = 1.0, 0.1
    n_c, n_t = sample_size_per_arm(sigma, eff, power=0.8)
    assert power_at(sigma, eff, n_c, n_t) == pytest.approx(0.8, abs=0.005)
    rng = np.random.default_rng(5)
    hits = [two_sample_test(rng.normal(0, 1, n_c), rng.normal(eff, 1, n_t)).p_value < 0.05 for _ in range(400)]
    assert np.mean(hits) == pytest.approx(0.8, abs=0.06)


def test_peeking_inflates_fpr_and_msprt_controls_it():
    rng = np.random.default_rng(6)
    y = (rng.random(200_000) < 0.05).astype(float)
    r = simulate_peeking(y, n_sims=300, n_per_sim=20_000, n_looks=20, seed=7)
    assert r["naive_peeking"] > 0.12
    assert r["msprt"] <= 0.06
    assert r["fixed"] < 0.09


def test_uplift_learners_find_signal():
    df = make_synthetic(n=150_000, seed=8)
    tr, te = df.iloc[:100_000], df.iloc[100_000:]
    X_tr, X_te = tr[FEATURES].values, te[FEATURES].values
    for name in ["T-learner", "X-learner"]:
        m = LEARNERS[name]().fit(X_tr, tr.treatment.values, tr.visit.values)
        s = m.predict(X_te)
        assert np.corrcoef(s, te.true_uplift)[0, 1] > 0.3, name
        assert qini_coefficient(te.visit.values, te.treatment.values, s) > 0.05, name


def test_cupac_beats_linear_on_nonlinear_outcome():
    df = make_synthetic(n=120_000, seed=9)
    X, t, y = df[FEATURES].values, df.treatment.values, df.visit.values
    lin = regression_adjustment(y, t, X)
    ml = cupac(y, t, X)
    assert ml.variance_reduction > lin.variance_reduction > 0


def test_aipw_removes_confounding_bias():
    from abkit.causal import aipw, cross_fit_nuisances, balance_table, propensity_diagnostics
    rng = np.random.default_rng(10)
    n = 120_000
    X = rng.normal(size=(n, 3))
    e_true = 1 / (1 + np.exp(-(1.7 + 0.8 * X[:, 0])))          # treated share depends on x0
    t = rng.random(n) < e_true
    p0 = 1 / (1 + np.exp(-(-3 + 1.0 * X[:, 0])))
    p1 = 1 / (1 + np.exp(-(-3 + 1.0 * X[:, 0] + 0.6 + 0.6 * X[:, 0])))
    y = (rng.random(n) < np.where(t, p1, p0)).astype(int)
    true_ate = (p1 - p0).mean()
    naive = y[t].mean() - y[~t].mean()
    e, mu = cross_fit_nuisances(X, t, {"y": y})
    r = aipw(y, t, e, mu["y"][1], mu["y"][0])
    assert abs(naive - true_ate) > 4 * r["se"]                 # naive is clearly biased
    assert abs(r["abs_lift"] - true_ate) < 3 * r["se"]         # AIPW recovers the truth
    assert propensity_diagnostics(t, e)["auc"] > 0.6
    assert abs(balance_table(X, t)["smd"].iloc[0]) > 0.1


def test_weighted_qini_equals_unweighted_with_unit_weights():
    rng = np.random.default_rng(11)
    y, t, s = rng.random(5000) < 0.1, rng.random(5000) < 0.5, rng.random(5000)
    assert qini_coefficient(y, t, s) == pytest.approx(qini_coefficient(y, t, s, weights=np.ones(5000)))


def test_qini_ties_not_broken_by_row_order():
    # all-tied scores + data sorted by treatment: a random ranking must look like random targeting
    n = 40_000
    t = np.r_[np.zeros(n // 2, bool), np.ones(n // 2, bool)]
    y = np.random.default_rng(12).random(n) < np.where(t, 0.06, 0.04)
    assert abs(qini_coefficient(y, t, np.zeros(n))) < 0.05
