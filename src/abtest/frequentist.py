"""Frequentist analysis: two-proportion z-test, Welch's t-test, confidence intervals,
power, and required sample size."""
from __future__ import annotations

import numpy as np
from scipy import stats


def two_proportion_ztest(control: np.ndarray, treat: np.ndarray) -> dict:
    nc, nt = len(control), len(treat)
    pc, pt = control.mean(), treat.mean()
    p_pool = (control.sum() + treat.sum()) / (nc + nt)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / nc + 1 / nt))
    z = (pt - pc) / se if se > 0 else 0.0
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    se_diff = np.sqrt(pc * (1 - pc) / nc + pt * (1 - pt) / nt)
    lo, hi = (pt - pc) - 1.96 * se_diff, (pt - pc) + 1.96 * se_diff
    return {"control_rate": round(float(pc), 5), "treat_rate": round(float(pt), 5),
            "abs_diff": round(float(pt - pc), 5),
            "rel_lift": round(float((pt - pc) / pc), 5) if pc else None,
            "z": round(float(z), 4), "p_value": round(float(p), 6),
            "ci95_abs_diff": [round(float(lo), 5), round(float(hi), 5)],
            "significant": bool(p < 0.05)}


def welch_ttest(control: np.ndarray, treat: np.ndarray) -> dict:
    t, p = stats.ttest_ind(treat, control, equal_var=False)
    diff = treat.mean() - control.mean()
    se = np.sqrt(treat.var(ddof=1) / len(treat) + control.var(ddof=1) / len(control))
    return {"mean_control": round(control.mean(), 4), "mean_treat": round(treat.mean(), 4),
            "diff": round(diff, 4), "t": round(float(t), 4), "p_value": round(float(p), 6),
            "ci95_diff": [round(diff - 1.96 * se, 4), round(diff + 1.96 * se, 4)],
            "significant": p < 0.05}


def required_sample_size(base_rate: float, mde_rel: float, alpha: float = 0.05,
                         power: float = 0.8) -> int:
    """Per-arm sample size to detect a relative MDE on a conversion rate."""
    p1 = base_rate
    p2 = base_rate * (1 + mde_rel)
    z_a = stats.norm.ppf(1 - alpha / 2)
    z_b = stats.norm.ppf(power)
    pbar = (p1 + p2) / 2
    n = (z_a * np.sqrt(2 * pbar * (1 - pbar)) + z_b * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p2 - p1) ** 2
    return int(np.ceil(n))


def achieved_power(n_per_arm: int, base_rate: float, mde_rel: float, alpha: float = 0.05) -> float:
    p1, p2 = base_rate, base_rate * (1 + mde_rel)
    se = np.sqrt(p1 * (1 - p1) / n_per_arm + p2 * (1 - p2) / n_per_arm)
    z_a = stats.norm.ppf(1 - alpha / 2)
    return float(1 - stats.norm.cdf(z_a - abs(p2 - p1) / se) + stats.norm.cdf(-z_a - abs(p2 - p1) / se))
