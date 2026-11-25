"""Experiment Analyzer — interactive A/B test + uplift dashboard.

Run locally:  streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from abkit import (LEARNERS, cupac, evaluate, regression_adjustment, sample_size_proportion, sample_size_per_arm,
                   sequential_path, simulate_peeking, srm_check, uplift_by_bins, qini_curve)
from abkit.causal import aipw, balance_table, cross_fit_nuisances, propensity_diagnostics
from abkit.data import FEATURES, make_synthetic

ROOT = Path(__file__).parent
DEMO = ROOT / "data" / "demo_sample.parquet"
RESULTS = ROOT / "results" / "results.json"
REPO_URL = "https://github.com/ericazhaoxy/ab-uplift-kit"  # <- update after pushing

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SERIES = [BLUE, ORANGE, AQUA, YELLOW]
MUTED, GRID = "#8f8e89", "#e6e5e1"

st.set_page_config(page_title="Experiment Analyzer", page_icon="🧪", layout="wide")


def style(fig, height=340):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(family="Inter, sans-serif", size=13),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), hovermode="closest")
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def pct(x, d=2):
    return f"{x * 100:.{d}f}%"


# ---------------------------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_demo():
    if DEMO.exists():
        return pd.read_parquet(DEMO), "Criteo Uplift v2.1 — random sample"
    return make_synthetic(300_000, seed=0).drop(columns="true_uplift"), "Synthetic Criteo-shaped data"


@st.cache_data(show_spinner=False)
def load_full_results():
    return json.loads(RESULTS.read_text()) if RESULTS.exists() else None


st.sidebar.title("🧪 Experiment Analyzer")
src = st.sidebar.radio("Data", ["Demo: Criteo ad experiment", "Upload my own CSV"])
if src.startswith("Demo"):
    df, label = load_demo()
    st.sidebar.caption(f"{label} · {len(df):,} users")
    treat_col, metrics, covs, exp_col = "treatment", ["visit", "conversion"], FEATURES, "exposure"
    design_share = 0.85
else:
    up = st.sidebar.file_uploader("CSV with one row per user", type="csv")
    st.sidebar.caption("Needs a 0/1 treatment column, one or more outcome columns, and optionally "
                       "pre-experiment covariates (used for variance reduction and uplift).")
    if up is None:
        st.info("Upload a CSV in the sidebar, or switch back to the demo dataset.")
        st.stop()
    df = pd.read_csv(up)
    label = up.name
    cols = list(df.columns)
    treat_col = st.sidebar.selectbox("Treatment column (0/1)", cols)
    num_cols = [c for c in cols if c != treat_col and pd.api.types.is_numeric_dtype(df[c])]
    metrics = st.sidebar.multiselect("Outcome metrics", num_cols, default=num_cols[:1])
    covs = st.sidebar.multiselect("Pre-experiment covariates", [c for c in num_cols if c not in metrics])
    exp_col = None
    design_share = st.sidebar.number_input("Designed treatment share", 0.01, 0.99, 0.5, 0.01)
    if not metrics:
        st.stop()

t = df[treat_col].values.astype(bool)
X = df[covs].values.astype(np.float32) if covs else None

st.title("Experiment Analyzer")
st.caption("Trustworthy A/B test readouts — SRM & balance guardrails, doubly-robust estimates, CUPED/CUPAC "
           "variance reduction, power planning, always-valid sequential tests and uplift targeting. "
           f"[Code on GitHub]({REPO_URL}) · built with `abkit`")

tabs = st.tabs(["① Health check", "② Effect estimates", "③ Power planner", "④ Peeking & sequential",
                "⑤ Uplift targeting", "⑥ Full-dataset results"])

# ---------------------------------------------------------------------------------------------
# 1. SRM
# ---------------------------------------------------------------------------------------------
with tabs[0]:
    st.subheader("Sample ratio mismatch (SRM) check")
    st.write("Before reading any metric, confirm the arms are the size the design promised. A mismatch "
             "usually means broken assignment or logging, and invalidates the results.")
    c1, c2 = st.columns([1, 1])
    with c1:
        share = st.number_input("Designed treatment share", 0.01, 0.99, float(design_share), 0.01, key="srm_share")
        bug = st.slider("Simulate a logging bug: % of control users silently dropped", 0.0, 2.0, 0.0, 0.1)
    rng = np.random.default_rng(0)
    keep = ~((~t) & (rng.random(len(t)) < bug / 100))
    n_c, n_t = int((~t & keep).sum()), int((t & keep).sum())
    r = srm_check([n_c, n_t], [1 - share, share])
    with c2:
        a, b, c = st.columns(3)
        a.metric("Control", f"{n_c:,}")
        b.metric("Treatment", f"{n_t:,}")
        c.metric("Observed share", pct(n_t / (n_c + n_t)))
    if r.srm_detected:
        st.error(f"⚠️ SRM detected — χ² = {r.chi2:.1f}, p = {r.p_value:.2e} (< {r.threshold}). "
                 "Do not trust the effect estimates until the cause is found.")
    else:
        st.success(f"✓ No SRM — χ² = {r.chi2:.2f}, p = {r.p_value:.3f}. Arm sizes match the design.")
    if exp_col and exp_col in df:
        st.caption(f"Exposure: {pct(df.loc[t, exp_col].mean())} of treated users actually saw an ad "
                   "(one-sided non-compliance). Effects below are intent-to-treat unless noted.")
    if X is not None:
        st.subheader("Covariate balance check")
        st.write("SRM only checks the *overall* split. If treatment probability varies with user features "
                 "(e.g. several tests with different ratios were pooled), a naive comparison is biased.")
        bal = balance_table(X, t, covs)
        chance = 1.96 * np.sqrt(1 / t.sum() + 1 / (~t).sum())
        b = bal.reindex(bal.smd.abs().sort_values().index)
        fig = go.Figure(go.Bar(x=b.smd, y=b.feature, orientation="h", marker_color=BLUE,
                               hovertemplate="%{y}: SMD %{x:.4f}<extra></extra>"))
        fig.add_vrect(x0=-chance, x1=chance, fillcolor=MUTED, opacity=0.25, line_width=0)
        fig.update_xaxes(title="Standardised mean difference (treatment − control)")
        c1, c2 = st.columns([3, 2])
        c1.plotly_chart(style(fig, 360), width="stretch")
        c1.caption("Gray band = 95% range of differences expected under pure randomisation.")
        n_bad = int((bal.z.abs() > 3.5).sum())
        if n_bad:
            c2.warning(f"⚠️ {n_bad} of {len(bal)} covariates differ between arms far beyond chance "
                       f"(max |z| = {bal.z.abs().max():.0f}). Treat assignment as *conditionally* random and "
                       "rely on the doubly-robust (AIPW) estimate in tab ②.")
        else:
            c2.success("✓ Covariates are balanced within chance.")

# ---------------------------------------------------------------------------------------------
# 2. Effect estimates
# ---------------------------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def estimate(key, metric, _y, _t, _X):
    y, t, X = _y, _t, _X
    from abkit import two_sample_test
    out = {"raw": two_sample_test(y[~t], y[t]).to_dict(), "lin": None, "aipw": None}
    if X is None:
        return out
    out["lin"] = regression_adjustment(y, t, X).adjusted.to_dict()
    e, mu = cross_fit_nuisances(X, t, {"y": y.astype(int) if set(np.unique(y)) <= {0, 1} else y})
    out["aipw"] = aipw(y, t, e, mu["y"][1], mu["y"][0])
    out["auc"] = propensity_diagnostics(t, e)["auc"]
    return out


with tabs[1]:
    st.subheader("Treatment effect with variance reduction")
    st.write("Pre-experiment covariates do two jobs: they **strip out noise** (CUPED / regression adjustment) and, "
             "when assignment isn't perfectly random, they **remove bias**. The doubly-robust **AIPW** estimator "
             "combines a propensity model and outcome models (both cross-fitted LightGBM) and stays unbiased if "
             "either one is right.")
    key = (label, len(df))
    results = {}
    with st.spinner("Fitting cross-fitted propensity and outcome models…"):
        for m in metrics:
            results[m] = estimate(key, m, df[m].values.astype(float), t, X)
    for m in metrics:
        res = results[m]
        st.markdown(f"#### {m}")
        raw = res["raw"]
        k = st.columns(4)
        k[0].metric("Control rate", pct(raw["mean_control"], 3))
        k[1].metric("Treatment rate", pct(raw["mean_treatment"], 3))
        best = res["aipw"] or raw
        k[2].metric("Relative lift (AIPW)" if res["aipw"] else "Relative lift", f"{best['rel_lift']:+.1%}",
                    help="Delta-method 95% CI shown in the chart")
        if res["aipw"]:
            k[3].metric("Naive estimate", f"{raw['rel_lift']:+.1%}",
                        delta=f"{(raw['rel_lift'] - best['rel_lift']) * 100:+.1f} pp bias", delta_color="off")
        rows = [("Naive diff-in-means", raw), ("Linear adjustment", res["lin"]), ("AIPW (doubly robust)", res["aipw"])]
        rows = [(n, r) for n, r in rows if r]
        fig = go.Figure()
        for i, (name, r) in enumerate(rows[::-1]):
            fig.add_trace(go.Scatter(
                x=[r["rel_ci_low"] * 100, r["rel_lift"] * 100, r["rel_ci_high"] * 100], y=[name] * 3,
                mode="lines+markers", line=dict(color=BLUE, width=3), marker=dict(size=[0, 12, 0], color=BLUE),
                showlegend=False,
                hovertemplate=f"<b>{name}</b><br>lift %{{x:.2f}}%<br>95% CI [{r['rel_ci_low'] * 100:.2f}%, "
                              f"{r['rel_ci_high'] * 100:.2f}%]<br>p = {r['p_value']:.2e}<extra></extra>"))
        fig.add_vline(x=0, line_color=MUTED, line_dash="dash")
        fig.update_xaxes(title="Relative lift vs control (%)")
        st.plotly_chart(style(fig, 230), width="stretch")
        if exp_col and exp_col in df and res["aipw"]:
            comp = df.loc[t, exp_col].mean()
            st.caption(f"Effect of actually seeing an ad (Wald/LATE): **{best['abs_lift'] / comp * 100:+.2f} pp** "
                       f"in {m} rate among exposed users (ITT {best['abs_lift'] * 100:+.3f} pp ÷ exposure {pct(comp)}).")

# ---------------------------------------------------------------------------------------------
# 3. Power planner
# ---------------------------------------------------------------------------------------------
with tabs[2]:
    st.subheader("How many users does the next experiment need?")
    m = st.selectbox("Metric", metrics, key="pw_metric")
    p0 = float(df.loc[~t, m].mean())
    binary = set(np.unique(df[m])) <= {0, 1}
    c1, c2, c3 = st.columns(3)
    mde_rel = c1.slider("Minimum detectable effect (relative)", 0.5, 20.0, 3.0, 0.5, format="%.1f%%") / 100
    share = c2.slider("Treatment share", 0.1, 0.9, 0.5, 0.05)
    power = c3.slider("Power", 0.7, 0.95, 0.8, 0.05)
    full = load_full_results()
    vr = (full or {}).get("aa_validation", {}).get(m, {}).get("variance_reduction_cupac", 0.3) if src.startswith("Demo") else 0.3
    vr = c1.slider("Variance reduction from CUPAC", 0.0, 0.9, float(round(max(vr, 0), 2)), 0.01,
                   help="Default for the demo = reduction measured in 1,000 A/A re-randomisations on the full data")
    sd = np.sqrt(p0 * (1 - p0)) if binary else float(df.loc[~t, m].std())
    ratio = share / (1 - share)
    n_raw = sum(sample_size_per_arm(sd, p0 * mde_rel, power=power, ratio=ratio))
    n_vr = sum(sample_size_per_arm(sd, p0 * mde_rel, power=power, ratio=ratio, variance_reduction=vr))
    a, b, c = st.columns(3)
    a.metric("Users needed — standard test", f"{n_raw:,}")
    b.metric("Users needed — with CUPAC", f"{n_vr:,}", delta=f"-{1 - n_vr / n_raw:.0%}", delta_color="inverse")
    c.metric("Baseline rate", pct(p0, 3))
    grid = np.linspace(0.01, 0.2, 60)
    fig = go.Figure()
    for name, v, col in [("Standard", 0.0, ORANGE), ("With CUPAC", vr, BLUE)]:
        ns = [sum(sample_size_per_arm(sd, p0 * g, power=power, ratio=ratio, variance_reduction=v)) for g in grid]
        fig.add_trace(go.Scatter(x=grid * 100, y=ns, name=name, line=dict(color=col, width=2),
                                 hovertemplate="MDE %{x:.1f}%<br>%{y:,.0f} users<extra>" + name + "</extra>"))
    fig.update_yaxes(type="log", title="Total users (log)")
    fig.update_xaxes(title="Relative MDE (%)")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(style(fig), width="stretch")

# ---------------------------------------------------------------------------------------------
# 4. Sequential
# ---------------------------------------------------------------------------------------------
with tabs[3]:
    st.subheader("Why you can't just check the dashboard every day")
    st.write("Re-computing a fixed-horizon p-value at every look and stopping at the first p < 0.05 inflates "
             "false positives. **mSPRT** always-valid p-values can be monitored continuously at the advertised "
             "error rate. Run A/A simulations on the real outcome distribution:")
    m = st.selectbox("Metric", metrics, key="seq_metric")
    c1, c2, c3 = st.columns(3)
    looks = c1.slider("Number of looks", 2, 50, 20)
    n_sim = c2.select_slider("Simulated A/A tests", [100, 200, 500, 1000], 200)
    n_per = c3.select_slider("Users per test", [20_000, 50_000, 100_000, 200_000], 50_000)
    if st.button("Run simulation", type="primary"):
        with st.spinner("Simulating…"):
            r = simulate_peeking(df.loc[~t, m].values.astype(float), n_sims=n_sim, n_per_sim=n_per, n_looks=looks)
        a, b, c = st.columns(3)
        a.metric("Fixed horizon (look once)", pct(r["fixed"], 1))
        b.metric(f"Naive peeking ({looks} looks)", pct(r["naive_peeking"], 1),
                 delta=f"{(r['naive_peeking'] - 0.05) * 100:+.1f} pp vs 5%", delta_color="inverse")
        c.metric(f"mSPRT ({looks} looks)", pct(r["msprt"], 1))
    st.markdown("##### This experiment, monitored as users arrive")
    n_slice = st.select_slider("Users in the monitored slice", [20_000, 50_000, 100_000, 200_000], 50_000)
    rs = np.random.default_rng(1).choice(len(df), size=min(n_slice, len(df)), replace=False)
    sp = sequential_path(df[m].values[rs], t[rs], n_looks=40)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sp["n"], y=sp["p_naive"], name="Fixed-horizon p (re-computed)",
                             line=dict(color=ORANGE, width=2)))
    fig.add_trace(go.Scatter(x=sp["n"], y=sp["p_msprt"], name="Always-valid p (mSPRT)", line=dict(color=BLUE, width=2)))
    fig.add_hline(y=0.05, line_dash="dash", line_color=MUTED, annotation_text="α = 0.05")
    fig.update_yaxes(type="log", title="p-value")
    fig.update_xaxes(title="Users observed")
    fig.update_layout(hovermode="x unified")
    st.plotly_chart(style(fig), width="stretch")

# ---------------------------------------------------------------------------------------------
# 5. Uplift
# ---------------------------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def fit_uplift(key, learners, _X, _t, _y):
    X, t, y = _X, _t, _y
    rng = np.random.default_rng(0)
    te = rng.random(len(y)) < 0.4
    out = {}
    for name in learners:
        m = LEARNERS[name]().fit(X[~te], t[~te], y[~te])
        out[name] = m.predict(X[te])
    return te, out


with tabs[4]:
    st.subheader("Who should get the treatment?")
    st.write("An uplift model predicts each user's **incremental** response. Targeting the most persuadable "
             "users captures most of the gain at a fraction of the cost.")
    if X is None:
        st.info("Select pre-experiment covariates in the sidebar to train uplift models.")
    else:
        m = st.selectbox("Outcome", metrics, key="up_metric")
        chosen = st.multiselect("Learners", list(LEARNERS), default=["T-learner", "X-learner"])
        if chosen:
            with st.spinner("Training uplift models…"):
                te, scores = fit_uplift((label, len(df), m), tuple(chosen), X, t, df[m].values)
            y_te, t_te = df[m].values[te], t[te]
            fig = go.Figure()
            evals = []
            for i, name in enumerate(chosen):
                frac, q = qini_curve(y_te, t_te, scores[name])
                fig.add_trace(go.Scatter(x=frac * 100, y=q / q[-1] * 100, name=name,
                                         line=dict(color=SERIES[i % 4], width=2)))
                evals.append(evaluate(name, y_te, t_te, scores[name]).to_dict())
            fig.add_trace(go.Scatter(x=[0, 100], y=[0, 100], name="Random", line=dict(color=MUTED, dash="dash")))
            fig.update_xaxes(title="% of users targeted (by predicted uplift)")
            fig.update_yaxes(title="% of incremental outcomes captured")
            fig.update_layout(hovermode="x unified")
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(style(fig, 380), width="stretch")
            ev = pd.DataFrame(evals).set_index("name")[["qini", "top10", "top20", "top30"]]
            ev.columns = ["Qini coef.", "Top 10%", "Top 20%", "Top 30%"]
            c2.markdown("**Share of incremental outcomes captured**")
            c2.dataframe(ev.style.format({"Qini coef.": "{:.3f}", "Top 10%": "{:.0%}", "Top 20%": "{:.0%}",
                                          "Top 30%": "{:.0%}"}), width="stretch")
            best = max(evals, key=lambda e: e["qini"])["name"]
            k = c2.slider("Target the top k% of users", 5, 100, 30, 5)
            from abkit import share_captured
            cap = share_captured(y_te, t_te, scores[best], k / 100)
            c2.success(f"With **{best}**, treating {k}% of users captures **{cap:.0%}** of incremental "
                       f"{m}s — {k}% of the treatment cost.")
            bins = uplift_by_bins(y_te, t_te, scores[best], 10)
            fig2 = go.Figure(go.Bar(x=bins["bin"], y=bins["observed_uplift"] * 100, marker_color=BLUE,
                                    error_y=dict(type="data", symmetric=False,
                                                 array=(bins["ci_high"] - bins["observed_uplift"]) * 100,
                                                 arrayminus=(bins["observed_uplift"] - bins["ci_low"]) * 100,
                                                 color=MUTED),
                                    hovertemplate="Decile %{x}<br>observed lift %{y:.3f} pp<extra></extra>"))
            fig2.update_xaxes(title="Predicted-uplift decile (1 = most persuadable)", dtick=1)
            fig2.update_yaxes(title="Observed lift (pp)")
            st.plotly_chart(style(fig2, 300), width="stretch")

# ---------------------------------------------------------------------------------------------
# 6. Full-dataset results
# ---------------------------------------------------------------------------------------------
with tabs[5]:
    res = load_full_results()
    if not res:
        st.info("Run `python scripts/run_analysis.py` to generate results on the full dataset.")
    else:
        st.subheader(f"Results on the full {res['source']} ({res['n_rows']:,} users)")
        v, cv = res["ate"]["visit"], res["ate"]["conversion"]
        aa = res["aa_validation"]["visit"]
        a, b, c, d = st.columns(4)
        a.metric("Visit lift (AIPW)", f"{v['aipw']['rel_lift']:+.1%}",
                 delta=f"naive {v['raw']['rel_lift']:+.1%}", delta_color="off")
        b.metric("Conversion lift (AIPW)", f"{cv['aipw']['rel_lift']:+.1%}",
                 delta=f"naive {cv['raw']['rel_lift']:+.1%}", delta_color="off")
        c.metric("CUPAC variance reduction (A/A)", pct(aa["variance_reduction_cupac"], 1),
                 delta=f"FPR {pct(aa['fpr_cupac'], 1)}", delta_color="off")
        d.metric("Peeking false-positive rate", pct(res["peeking"]["aa"]["naive_peeking"], 1),
                 delta=f"mSPRT {pct(res['peeking']['aa']['msprt'], 1)}", delta_color="off")
        if "uplift" in res:
            best = next(m for m in res["uplift"]["models"] if m["name"] == res["uplift"]["best"])
            st.markdown(f"**Uplift:** {best['name']} captures **{best['top20']:.0%}** of incremental visits by "
                        f"targeting the top 20% of {res['uplift']['n_test']:,} held-out users.")
            st.dataframe(pd.DataFrame(res["uplift"]["models"]).set_index("name"), width="stretch")
        for f in ["balance", "ate_estimators", "peeking_fpr", "sequential_path", "qini_curves", "uplift_deciles"]:
            p = ROOT / "results" / "figures" / f"{f}.png"
            if p.exists():
                st.image(str(p))
