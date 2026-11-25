"""Data loading: the Criteo Uplift v2.1 dataset and a synthetic look-alike for tests/demos.

Criteo Uplift v2.1 (Diemert et al., 2018/2021): ~13.98M users from a randomised
incrementality test of an ad campaign.
  f0..f11     12 anonymised pre-treatment features (dense floats)
  treatment   1 = eligible to see ads (~85%), 0 = held-out control (~15%)
  exposure    1 = actually saw an ad (only possible if treated)
  visit       1 = visited the advertiser's site
  conversion  1 = converted
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = [f"f{i}" for i in range(12)]
CRITEO_FILE = "criteo-research-uplift-v2.1.csv.gz"
# The official link (go.criteo.net) now redirects to a dead host; this mirror serves the
# identical file (SHA-256 below matches the checksum published with the original).
CRITEO_URL = "https://github.com/jszymon/uplift_sklearn_data/releases/download/Criteo/" + CRITEO_FILE
CRITEO_SHA256 = "2716e1bf0fd157a93b5bf86924d9088419dfbac2022c6cd90030220634f616dc"
DTYPES = {**{f: "float32" for f in FEATURES}, "treatment": "int8", "conversion": "int8", "visit": "int8",
          "exposure": "int8"}


def load_criteo(data_dir: str | Path = "data", sample_frac: float | None = None, seed: int = 42) -> pd.DataFrame:
    """Load the full dataset (cached as parquet after the first read), optionally subsampled."""
    data_dir = Path(data_dir)
    parquet = data_dir / "criteo_uplift.parquet"
    if parquet.exists():
        df = pd.read_parquet(parquet)
    else:
        csv = data_dir / CRITEO_FILE
        if not csv.exists():
            raise FileNotFoundError(f"{csv} not found. Run: curl -L -o {csv} {CRITEO_URL}")
        df = pd.read_csv(csv, dtype=DTYPES)
        df.to_parquet(parquet, index=False)
    if sample_frac is not None and sample_frac < 1:
        df = df.sample(frac=sample_frac, random_state=seed).reset_index(drop=True)
    return df


def make_synthetic(n: int = 500_000, treat_share: float = 0.85, seed: int = 0) -> pd.DataFrame:
    """Criteo-shaped synthetic data with a known, heterogeneous treatment effect.

    Base visit rate ~4%, average lift ~ +1pp concentrated in a 'persuadable'
    segment, so uplift models have real signal to find. Column ``true_uplift``
    holds the ground-truth individual effect on visit probability.
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 12)).astype("float32")
    X[:, 3] = np.abs(X[:, 3])  # a few skewed, count-like features
    X[:, 7] = rng.exponential(1.0, n)
    t = (rng.random(n) < treat_share).astype("int8")

    logit_base = -3.6 + 1.1 * X[:, 0] - 0.6 * X[:, 1] + 0.5 * X[:, 2] * X[:, 3] + 0.4 * np.sin(2 * X[:, 4])
    p0 = 1 / (1 + np.exp(-logit_base))
    persuadable = 1 / (1 + np.exp(-(1.5 * X[:, 5] + 1.0 * X[:, 6] - 0.5)))
    lift_logit = 0.9 * persuadable - 0.15  # small negative effect for 'sleeping dogs'
    p1 = 1 / (1 + np.exp(-(logit_base + lift_logit)))
    visit = (rng.random(n) < np.where(t == 1, p1, p0)).astype("int8")
    conv_p = 1 / (1 + np.exp(-(-2.6 + 0.6 * X[:, 0] + 0.3 * X[:, 8])))
    conversion = (visit & (rng.random(n) < conv_p)).astype("int8")
    exposure = (t & (rng.random(n) < 0.036 * (1 + X[:, 7]))).astype("int8")

    df = pd.DataFrame(X, columns=FEATURES)
    df["treatment"] = t
    df["conversion"] = conversion
    df["visit"] = visit
    df["exposure"] = exposure
    df["true_uplift"] = (p1 - p0).astype("float32")
    return df
