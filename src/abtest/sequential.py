"""Peeking-safe sequential testing via mSPRT (mixture Sequential Probability Ratio Test).

Repeatedly running a fixed-horizon z-test and stopping when p<0.05 inflates the false-positive
rate badly (the "peeking problem"). An always-valid mSPRT p-value can be monitored continuously
without inflation. `peeking_demo` quantifies both on null data.
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def msprt_pvalue(d: np.ndarray, tau2: float = 1e-3) -> float:
    """Always-valid p-value for H0: mean(d)=0, via a normal-mixture SPRT (running-min safe)."""
    n = len(d)
    if n < 2:
        return 1.0
    dbar = d.mean()
    s2 = max(d.var(ddof=1), 1e-6)
    lam = np.sqrt(s2 / (s2 + n * tau2)) * np.exp((n ** 2 * tau2 * dbar ** 2) / (2 * s2 * (s2 + n * tau2)))
    return min(1.0, 1.0 / lam)


def _naive_p(control: np.ndarray, treat: np.ndarray) -> float:
    pc, pt = control.mean(), treat.mean()
    nc, nt = len(control), len(treat)
    pp = (control.sum() + treat.sum()) / (nc + nt)
    se = np.sqrt(pp * (1 - pp) * (1 / nc + 1 / nt))
    if se == 0:
        return 1.0
    z = (pt - pc) / se
    return 2 * (1 - stats.norm.cdf(abs(z)))


def peeking_demo(n_per_arm: int = 2000, base_rate: float = 0.12, n_looks: int = 10,
                 n_sims: int = 300, alpha: float = 0.05, seed: int = 7) -> dict:
    """On NULL data (no real effect), measure the false-positive rate of naive repeated
    peeking vs the always-valid mSPRT. Naive should blow past alpha; mSPRT should not."""
    rng = np.random.default_rng(seed)
    looks = np.linspace(n_per_arm // n_looks, n_per_arm, n_looks).astype(int)
    naive_fp = av_fp = 0
    for _ in range(n_sims):
        c = rng.binomial(1, base_rate, n_per_arm)
        t = rng.binomial(1, base_rate, n_per_arm)        # same rate -> null is true
        d = (t - c).astype(float)
        naive_hit = any(_naive_p(c[:m], t[:m]) < alpha for m in looks)
        av_hit = min(msprt_pvalue(d[:m]) for m in looks) < alpha
        naive_fp += naive_hit
        av_fp += av_hit
    return {"alpha": alpha, "n_looks": n_looks, "n_sims": n_sims,
            "naive_peeking_fp_rate": round(naive_fp / n_sims, 4),
            "always_valid_fp_rate": round(av_fp / n_sims, 4)}
