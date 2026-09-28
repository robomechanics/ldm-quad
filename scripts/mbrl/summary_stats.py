"""Shared statistics for the Level 2 / Level 3 summaries: 95% t confidence intervals over seeds.

T975[df] = two-sided 95% Student-t critical value (scipy.stats.t.ppf(0.975, df)), df = 1..30;
beyond 30 the normal 1.96 is used (the error is < 2%).
"""

from __future__ import annotations

import math
import statistics

T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def t975(df: int) -> float:
    return T975.get(df, 1.96)


def mean_ci(xs) -> tuple[float, float]:
    """(mean, 95% CI half-width) over the non-NaN values; half-width inf for n=1, NaNs for n=0."""
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return math.nan, math.nan
    m = statistics.fmean(xs)
    if len(xs) < 2:
        return m, math.inf
    return m, t975(len(xs) - 1) * statistics.stdev(xs) / math.sqrt(len(xs))


def fmt(m: float, ci: float, spec: str = "{:.4f}") -> str:
    """'m+-ci'; '   nan' / '  >run' for NaN / inf means, '+-inf' for n=1."""
    if math.isnan(m):
        return "   nan"
    if math.isinf(m):
        return "  >run"
    return spec.format(m) + ("" if math.isnan(ci) else ("+-inf" if math.isinf(ci) else "+-" + spec.format(ci)))
