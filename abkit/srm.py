"""Sample Ratio Mismatch (SRM) check.

An SRM means the observed split between arms differs from the designed split
more than chance allows. It almost always signals a bug (broken randomisation,
logging loss, bot filtering applied to one arm...) and invalidates the test.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Sequence

import numpy as np
from scipy import stats


@dataclass
class SRMResult:
    observed: list
    expected: list
    expected_ratio: list
    chi2: float
    p_value: float
    threshold: float
    srm_detected: bool

    def to_dict(self) -> dict:
        return asdict(self)


def srm_check(counts: Sequence[int], expected_ratio: Sequence[float], threshold: float = 0.001) -> SRMResult:
    """Chi-square goodness-of-fit test on arm sizes.

    A strict threshold (default p < 0.001) is industry practice because the
    check runs on every experiment and false alarms are costly.
    """
    counts = np.asarray(counts, dtype=float)
    ratio = np.asarray(expected_ratio, dtype=float)
    ratio = ratio / ratio.sum()
    expected = counts.sum() * ratio
    chi2, p = stats.chisquare(counts, expected)
    return SRMResult(
        observed=counts.astype(int).tolist(),
        expected=expected.round(1).tolist(),
        expected_ratio=ratio.tolist(),
        chi2=float(chi2),
        p_value=float(p),
        threshold=threshold,
        srm_detected=bool(p < threshold),
    )
