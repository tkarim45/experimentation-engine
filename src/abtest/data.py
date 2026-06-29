"""Simulate experiments with a known ground-truth effect — so the methods can be validated
(a real lift should be detected; a null should not). Supports binary (conversion) and
continuous metrics, with an optional pre-experiment covariate for CUPED.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Experiment:
    # binary outcomes
    control_conv: np.ndarray          # 0/1 per user
    treat_conv: np.ndarray
    # continuous metric + correlated pre-experiment covariate (for CUPED)
    control_metric: np.ndarray
    treat_metric: np.ndarray
    control_pre: np.ndarray
    treat_pre: np.ndarray
    true_lift: float                   # ground-truth relative lift on conversion


def simulate(n_per_arm: int = 5000, base_rate: float = 0.12, rel_lift: float = 0.10,
             seed: int = 7) -> Experiment:
    rng = np.random.default_rng(seed)
    p_c, p_t = base_rate, base_rate * (1 + rel_lift)
    cc = rng.binomial(1, p_c, n_per_arm)
    tc = rng.binomial(1, p_t, n_per_arm)

    # continuous revenue-like metric, correlated with a pre-period covariate (ρ≈0.7)
    pre_c = rng.normal(50, 15, n_per_arm)
    pre_t = rng.normal(50, 15, n_per_arm)
    noise_c = rng.normal(0, 10, n_per_arm)
    noise_t = rng.normal(0, 10, n_per_arm)
    mc = 0.7 * pre_c + noise_c + 20
    mt = 0.7 * pre_t + noise_t + 20 + 2.0 * rel_lift / 0.10   # small additive treat effect
    return Experiment(cc, tc, mc, mt, pre_c, pre_t, true_lift=rel_lift)
