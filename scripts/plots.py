"""Static figures for the README (matplotlib)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT, TEXT2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8f8e89", "#e6e5e1", "#fcfcfb"

_family = "Inter" if any("Inter" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams.update({
    "font.family": _family, "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlecolor": TEXT, "axes.titlelocation": "left",
    "savefig.dpi": 160, "savefig.bbox": "tight",
})


def _pct(x, digits=1):
    return f"{x * 100:.{digits}f}%"


def estimator_plot(ate: dict, path):
    """Forest plot: relative lift and 95% CI under each estimator, one panel per metric."""
    methods = [("raw", "Naive diff-in-means"), ("regression_adjustment", "Linear adjustment"),
               ("ipw", "IPW"), ("aipw", "AIPW (doubly robust)")]
    fig, axes = plt.subplots(1, len(ate), figsize=(5.4 * len(ate), 2.9), sharey=True)
    for ax, (metric, res) in zip(np.atleast_1d(axes), ate.items()):
        for i, (key, label) in enumerate(methods):
            r = res[key]
            y = len(methods) - 1 - i
            col = SERIES[0] if key == "aipw" else MUTED
            lo = r.get("rel_ci_low")
            hi = r.get("rel_ci_high")
            if lo is not None:
                ax.plot([lo * 100, hi * 100], [y, y], color=col, lw=2, solid_capstyle="round")
            ax.plot(r["rel_lift"] * 100, y, "o", ms=8, color=col, mec=SURFACE, mew=2)
            ax.text(r["rel_lift"] * 100, y + 0.25, f"{r['rel_lift'] * 100:+.1f}%", color=TEXT2, fontsize=9,
                    ha="center")
        ax.set_title(f"{metric.capitalize()}: relative lift by estimator")
        ax.set_yticks(range(len(methods)))
        ax.set_yticklabels([m[1] for m in methods][::-1])
        ax.set_xlabel("Relative lift, treatment vs control (%)")
        ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(methods) - 0.2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def balance_plot(bal, path):
    """Standardised mean differences per feature, with the band expected from chance."""
    b = bal.reindex(bal.smd.abs().sort_values().index)
    chance = 1.96 * (b.smd / b.z).abs().iloc[0]
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.barh(b.feature, b.smd, color=SERIES[0], height=0.6, edgecolor=SURFACE, linewidth=1.5)
    ax.axvspan(-chance, chance, color=MUTED, alpha=0.25, lw=0)
    ax.axvline(0, color=TEXT2, lw=0.8)
    ax.set_xlabel("Standardised mean difference (treatment - control)")
    ax.set_title("Covariate balance: far outside the chance band")
    ax.text(0.98, 0.04, "gray band = 95% range\nunder pure randomisation", transform=ax.transAxes, color=TEXT2,
            fontsize=8.5, ha="right", va="bottom")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def peeking_plot(peek: dict, path):
    aa = peek["aa"]
    labels = ["Fixed horizon\n(look once)", "Naive peeking\n(20 looks)", "mSPRT\n(20 looks)"]
    vals = [aa["fixed"], aa["naive_peeking"], aa["msprt"]]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    bars = ax.bar(labels, [v * 100 for v in vals], color=SERIES[0], width=0.55, edgecolor=SURFACE, linewidth=2)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.6, _pct(v), ha="center", color=TEXT, fontsize=10,
                bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    ax.axhline(5, color=MUTED, lw=1.2, ls="--")
    ax.text(2.42, 5.3, "5% target", color=TEXT2, fontsize=9, ha="right")
    ax.set_ylabel("False-positive rate (A/A)")
    ax.set_title(f"Peeking inflates false positives ({aa['n_sims']:,} A/A tests)")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, max(vals) * 100 * 1.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def sequential_plot(sp: dict, path):
    n = np.array(sp["n"]) / 1000
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.plot(n, sp["p_naive"], color=SERIES[1], lw=2, label="Fixed-horizon p-value (re-computed each look)")
    ax.plot(n, sp["p_msprt"], color=SERIES[0], lw=2, label="Always-valid p-value (mSPRT)")
    ax.axhline(0.05, color=MUTED, lw=1.2, ls="--")
    ax.set_yscale("log")
    ax.set_xlabel("Users observed (thousands)")
    ax.set_ylabel("p-value (log scale)")
    ax.set_title("Monitoring the Criteo test as data arrives")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def qini_plot(curves: dict, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    last = None
    for i, (name, (frac, q)) in enumerate(curves.items()):
        qn = q / q[-1]
        ax.plot(frac * 100, qn * 100, color=SERIES[i % len(SERIES)], lw=2, label=name)
        last = frac
    ax.plot([0, 100], [0, 100], color=MUTED, lw=1.2, ls="--", label="Random targeting")
    ax.set_xlabel("Share of users targeted, ranked by predicted uplift (%)")
    ax.set_ylabel("Share of incremental visits captured (%)")
    ax.set_title("Qini curves on held-out users (IPW-weighted)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    ax.set_xlim(0, 100)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def decile_plot(bins, model_name, path):
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = bins["bin"].values
    y = bins["observed_uplift"].values * 100
    lo, hi = bins["ci_low"].values * 100, bins["ci_high"].values * 100
    ax.bar(x, y, color=SERIES[0], width=0.7, edgecolor=SURFACE, linewidth=2)
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="none", ecolor=TEXT2, elinewidth=1, capsize=3)
    ax.axhline(0, color=TEXT2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xlabel("Predicted-uplift decile (1 = most persuadable)")
    ax.set_ylabel("Observed lift in visit rate (pp)")
    ax.set_title(f"Observed treatment effect by {model_name} decile")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
