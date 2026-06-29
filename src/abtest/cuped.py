"""CUPED (Controlled-experiment Using Pre-Experiment Data) — variance reduction.

Subtract the part of the metric explained by a pre-experiment covariate. Same unbiased
treatment effect, smaller variance → tighter CIs and more power for free.
"""
from __future__ import annotations

import numpy as np

from .frequentist import welch_ttest


def cuped_adjust(metric: np.ndarray, pre: np.ndarray, theta: float | None = None):
    if theta is None:
        theta = np.cov(metric, pre, ddof=1)[0, 1] / np.var(pre, ddof=1)
    adjusted = metric - theta * (pre - pre.mean())
    return adjusted, float(theta)


def apply(control_metric, control_pre, treat_metric, treat_pre) -> dict:
    """Estimate theta on the pooled data, adjust both arms, and compare the t-test
    before vs after — reporting the variance reduction and the tighter CI."""
    pooled_m = np.concatenate([control_metric, treat_metric])
    pooled_p = np.concatenate([control_pre, treat_pre])
    theta = np.cov(pooled_m, pooled_p, ddof=1)[0, 1] / np.var(pooled_p, ddof=1)

    c_adj, _ = cuped_adjust(control_metric, control_pre, theta)
    t_adj, _ = cuped_adjust(treat_metric, treat_pre, theta)

    before = welch_ttest(control_metric, treat_metric)
    after = welch_ttest(c_adj, t_adj)
    var_before = np.var(pooled_m, ddof=1)
    var_after = np.var(np.concatenate([c_adj, t_adj]), ddof=1)
    return {
        "theta": round(theta, 4),
        "variance_reduction": round(1 - var_after / var_before, 4),
        "ci_width_before": round(before["ci95_diff"][1] - before["ci95_diff"][0], 4),
        "ci_width_after": round(after["ci95_diff"][1] - after["ci95_diff"][0], 4),
        "p_before": before["p_value"],
        "p_after": after["p_value"],
    }
