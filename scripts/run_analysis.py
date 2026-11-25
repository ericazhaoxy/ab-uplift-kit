"""End-to-end analysis of the Criteo Uplift experiment.

Usage
-----
    python scripts/run_analysis.py                    # full Criteo dataset in data/
    python scripts/run_analysis.py --sample-frac 0.1  # quicker run on 10%
    python scripts/run_analysis.py --synthetic        # no download needed (for CI / smoke tests)

Writes results/results.json, results/*.csv and results/figures/*.png.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from abkit import (LEARNERS, evaluate, regression_adjustment, sample_size_proportion,  # noqa: E402
                   sequential_path, simulate_peeking, srm_check, two_sample_test, uplift_by_bins, qini_curve)
from abkit.causal import aipw, balance_table, cross_fit_nuisances, ipw, propensity_diagnostics  # noqa: E402
from abkit.data import FEATURES, load_criteo, make_synthetic  # noqa: E402
from scripts import plots  # noqa: E402

RES = ROOT / "results"
FIG = RES / "figures"


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def aa_validation(y, pred, n_sims=1000, n=500_000, treat_share=0.85, seed=0):
    """Re-randomise real outcomes (true effect = 0, assignment truly random) many times.

    Measures, for CUPAC with an ML prediction ``pred`` of y from pre-treatment features:
      - false-positive rate (should be ~5%) and CI coverage (should be ~95%)
      - average variance reduction vs the unadjusted test
    """
    rng = np.random.default_rng(seed)
    rej_raw = rej_adj = cover = 0
    ratios = []
    for _ in range(n_sims):
        idx = rng.integers(0, len(y), n)
        yy, pp = y[idx], pred[idx]
        t = rng.random(n) < treat_share
        theta = np.cov(yy, pp)[0, 1] / pp.var()
        ya = yy - theta * (pp - pp.mean())
        r_raw = two_sample_test(yy[~t], yy[t])
        r_adj = two_sample_test(ya[~t], ya[t])
        rej_raw += r_raw.p_value < 0.05
        rej_adj += r_adj.p_value < 0.05
        cover += r_adj.abs_ci_low <= 0 <= r_adj.abs_ci_high
        ratios.append((r_adj.se / r_raw.se) ** 2)
    return {"n_sims": n_sims, "n_per_sim": n, "fpr_raw": rej_raw / n_sims, "fpr_cupac": rej_adj / n_sims,
            "ci_coverage_cupac": cover / n_sims, "variance_reduction_cupac": float(1 - np.mean(ratios))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--sample-frac", type=float, default=None)
    ap.add_argument("--uplift-train-n", type=int, default=3_000_000)
    ap.add_argument("--aa-sims", type=int, default=1000)
    ap.add_argument("--peek-sims", type=int, default=2000)
    ap.add_argument("--skip-uplift", action="store_true")
    ap.add_argument("--seq-n", type=int, default=80_000, help="users in the monitored-experiment slice")
    args = ap.parse_args()
    global FIG
    if args.synthetic:  # never overwrite the real-data figures
        FIG = RES / "figures_synthetic"
    FIG.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    if args.synthetic:
        df = make_synthetic(n=1_000_000, seed=0)
        source = "synthetic (Criteo-shaped)"
    else:
        df = load_criteo(ROOT / "data", sample_frac=args.sample_frac)
        source = "Criteo Uplift v2.1" + (f" ({args.sample_frac:.0%} sample)" if args.sample_frac else "")
    log(f"Loaded {len(df):,} rows from {source} in {time.time() - t0:.0f}s")
    out: dict = {"source": source, "n_rows": int(len(df))}

    t = df["treatment"].values.astype(bool)
    X = df[FEATURES].values.astype(np.float32)
    ys = {m: df[m].values.astype(np.int8) for m in ["visit", "conversion"]}
    compliance = float(df.loc[t, "exposure"].mean())
    out["overview"] = {
        "n_treatment": int(t.sum()), "n_control": int((~t).sum()), "treat_share": float(t.mean()),
        "exposure_rate_treated": compliance,
        **{f"{m}_rate_{arm}": float(ys[m][mask].mean()) for m in ys for arm, mask in [("control", ~t), ("treatment", t)]},
    }

    # 1. Health checks: SRM + covariate balance -----------------------------------------
    srm = srm_check([int((~t).sum()), int(t.sum())], [0.15, 0.85])
    out["srm"] = srm.to_dict()
    rng = np.random.default_rng(1)
    drop = (~t) & (rng.random(len(t)) < 0.005)  # what a 0.5% logging loss in control looks like
    out["srm_injected_bug"] = srm_check([int((~t & ~drop).sum()), int(t.sum())], [0.15, 0.85]).to_dict()
    bal = balance_table(X, t, FEATURES)
    bal.to_csv(RES / ("balance_synthetic.csv" if args.synthetic else "balance.csv"), index=False)
    out["balance"] = {"max_abs_smd": float(bal.smd.abs().max()), "max_abs_z": float(bal.z.abs().max()),
                      "chance_smd_scale": float(np.sqrt(1 / t.sum() + 1 / (~t).sum())),
                      "table": bal.round(5).to_dict(orient="records")}
    log(f"SRM p={srm.p_value:.3g}; max |SMD|={out['balance']['max_abs_smd']:.4f} "
        f"(max |z|={out['balance']['max_abs_z']:.0f})")

    # 2. Cross-fitted nuisance models: propensity e(x), outcome models mu1(x), mu0(x) ------
    tic = time.time()
    cache = ROOT / "data" / f"nuisances_{len(df)}.npz"
    if cache.exists() and not args.synthetic:
        z = np.load(cache)
        e, mu = z["e"], {m: (z[f"{m}_mu0"], z[f"{m}_mu1"]) for m in ys}
    else:
        e, mu = cross_fit_nuisances(X, t, ys)
        if not args.synthetic:
            np.savez_compressed(cache, e=e, **{f"{m}_mu{a}": mu[m][a] for m in ys for a in (0, 1)})
    out["propensity"] = propensity_diagnostics(t, e)
    log(f"Nuisances fitted in {time.time() - tic:.0f}s; propensity AUC={out['propensity']['auc']:.4f}")

    # 3. Average treatment effects ---------------------------------------------------------
    out["ate"] = {}
    for m, y in ys.items():
        yf = y.astype(np.float64)
        raw = two_sample_test(yf[~t], yf[t], metric=m).to_dict()
        lin = regression_adjustment(yf, t, X, metric=m).adjusted.to_dict()
        dr = aipw(yf, t, e, mu[m][1], mu[m][0])
        w = ipw(yf, t, e)
        out["ate"][m] = {"raw": raw, "regression_adjustment": lin, "ipw": w, "aipw": dr,
                         "late_exposed": {"late": dr["abs_lift"] / compliance, "ci_low": dr["abs_ci_low"] / compliance,
                                          "ci_high": dr["abs_ci_high"] / compliance, "compliance": compliance},
                         "bias_naive_vs_aipw": raw["abs_lift"] - dr["abs_lift"],
                         "se_ratio_aipw_vs_raw": dr["se"] / raw["se"]}
        log(f"{m}: naive {raw['rel_lift']:+.1%} | linear {lin['rel_lift']:+.1%} | IPW {w['rel_lift']:+.1%} | "
            f"AIPW {dr['rel_lift']:+.1%} [{dr['rel_ci_low']:+.1%}, {dr['rel_ci_high']:+.1%}]")

    # 4. A/A validation + variance reduction of CUPAC under clean randomisation ----------------
    out["aa_validation"] = {}
    for m, y in ys.items():
        pred = e * mu[m][1] + (1 - e) * mu[m][0]  # cross-fitted prediction of y from x alone
        out["aa_validation"][m] = aa_validation(y.astype(float), pred, n_sims=args.aa_sims, n=min(500_000, len(df)))
        log(f"A/A {m}: {out['aa_validation'][m]}")

    # 5. Power / sample-size planning ---------------------------------------------------------
    out["power"] = {}
    for m, mde_rel in [("visit", 0.02), ("conversion", 0.05)]:
        p = out["overview"][f"{m}_rate_control"]
        vr = out["aa_validation"][m]["variance_reduction_cupac"]
        rows = {}
        for name, v in [("raw", 0.0), ("cupac", max(vr, 0.0))]:
            n_c, n_t = sample_size_proportion(p, mde_rel, ratio=0.85 / 0.15, variance_reduction=v)
            rows[name] = {"n_control": n_c, "n_treatment": n_t, "n_total": n_c + n_t}
        out["power"][m] = {"mde_rel": mde_rel, "baseline": p, "variance_reduction": vr, **rows}

    # 6. Sequential testing --------------------------------------------------------------------
    y_ctrl = ys["visit"][~t].astype(float)
    peek_aa = simulate_peeking(y_ctrl, n_sims=args.peek_sims, n_per_sim=200_000, n_looks=20, treat_share=0.5)
    peek_ab = simulate_peeking(y_ctrl, n_sims=args.peek_sims // 2, n_per_sim=200_000, n_looks=20,
                               treat_share=0.5, effect=y_ctrl.mean() * 0.05, seed=1)
    out["peeking"] = {"aa": peek_aa, "ab_5pct_lift": peek_ab}
    log(f"Peeking A/A: fixed {peek_aa['fixed']:.3f} naive {peek_aa['naive_peeking']:.3f} msprt {peek_aa['msprt']:.3f}")
    rs = np.random.default_rng(2).choice(len(df), size=min(args.seq_n, len(df)), replace=False)
    path = sequential_path(ys["visit"][rs], t[rs], n_looks=40)
    out["sequential_path"] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in path.items()}

    # 7. Uplift modelling ----------------------------------------------------------------------
    if not args.skip_uplift:
        rng = np.random.default_rng(3)
        is_test = rng.random(len(df)) < 0.3
        tr_idx = np.flatnonzero(~is_test)
        if len(tr_idx) > args.uplift_train_n:
            tr_idx = rng.choice(tr_idx, args.uplift_train_n, replace=False)
        te_idx = np.flatnonzero(is_test)
        y = ys["visit"]
        # Trimmed propensities for weighting: a handful of users with e(x) ~ 0.99 would otherwise get
        # control weights of ~100 and dominate the curve (affects <1% of users).
        e_w = np.clip(e, 0.05, 0.95)
        w_te = np.where(t[te_idx], 1 / e_w[te_idx], 1 / (1 - e_w[te_idx]))  # IPW weights for evaluation
        evals, curves, scores = [], {}, {}
        for name, cls in LEARNERS.items():
            tic = time.time()
            if name == "Transformed outcome":
                model = cls().fit(X[tr_idx], t[tr_idx], y[tr_idx], e=e_w[tr_idx])
            else:
                model = cls().fit(X[tr_idx], t[tr_idx], y[tr_idx])
            fit_s = time.time() - tic
            s = model.predict(X[te_idx])
            scores[name] = s
            ev = evaluate(name, y[te_idx], t[te_idx], s, fit_s).to_dict()
            evw = evaluate(name, y[te_idx], t[te_idx], s, weights=w_te).to_dict()
            ev.update(qini_ipw=evw["qini"], top10_ipw=evw["top10"], top20_ipw=evw["top20"], top30_ipw=evw["top30"])
            evals.append(ev)
            curves[name] = qini_curve(y[te_idx], t[te_idx], s, weights=w_te)
            log(f"{name}: qini={ev['qini']:.3f} (IPW {ev['qini_ipw']:.3f}) top20={ev['top20']:.1%} fit={fit_s:.0f}s")
        best = max(evals, key=lambda d: d["qini_ipw"])["name"]
        bins = uplift_by_bins(y[te_idx], t[te_idx], scores[best], n_bins=10)
        bins.to_csv(RES / ("uplift_deciles_synthetic.csv" if args.synthetic else "uplift_deciles.csv"), index=False)
        out["uplift"] = {"target": "visit", "n_train": int(len(tr_idx)), "n_test": int(len(te_idx)),
                         "models": evals, "best": best, "deciles": bins.to_dict(orient="records")}
        if "true_uplift" in df:
            out["uplift"]["corr_true"] = {k: float(np.corrcoef(v, df["true_uplift"].values[te_idx])[0, 1])
                                          for k, v in scores.items()}
        plots.qini_plot(curves, FIG / "qini_curves.png")
        plots.decile_plot(bins, best, FIG / "uplift_deciles.png")

    # Figures ----------------------------------------------------------------------------------
    plots.estimator_plot(out["ate"], FIG / "ate_estimators.png")
    plots.peeking_plot(out["peeking"], FIG / "peeking_fpr.png")
    plots.sequential_plot(out["sequential_path"], FIG / "sequential_path.png")
    plots.balance_plot(bal, FIG / "balance.png")

    out["runtime_seconds"] = time.time() - t0
    suffix = "_synthetic" if args.synthetic else ""
    (RES / f"results{suffix}.json").write_text(json.dumps(out, indent=2, default=float))
    log(f"Done in {out['runtime_seconds'] / 60:.1f} min -> results/results{suffix}.json")


if __name__ == "__main__":
    main()
