"""Create the small random sample of Criteo that ships with the repo and powers the hosted demo.

    python scripts/make_demo_sample.py --n 300000
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from abkit.data import load_criteo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300_000)
args = ap.parse_args()

df = load_criteo(ROOT / "data")
sample = df.sample(n=args.n, random_state=7).reset_index(drop=True)
out = ROOT / "data" / "demo_sample.parquet"
sample.to_parquet(out, index=False, compression="zstd")
print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(sample):,} rows)")
