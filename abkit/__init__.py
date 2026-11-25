"""abkit — experimentation & uplift-modelling toolkit."""
from .srm import srm_check, SRMResult
from .stats import two_sample_test, TestResult
from .variance_reduction import cuped, cupac, regression_adjustment, VarianceReductionResult
from .power import sample_size_per_arm, sample_size_proportion, mde, power_at
from .sequential import always_valid_pvalues, sequential_path, simulate_peeking
from .uplift import LEARNERS, qini_curve, qini_coefficient, share_captured, uplift_by_bins, evaluate

__version__ = "0.1.0"
