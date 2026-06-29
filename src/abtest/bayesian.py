"""Bayesian analysis for conversion experiments: Beta-Binomial posteriors, P(treat > control),
expected loss of each decision, and credible intervals. Answers the question stakeholders
actually ask — "what's the probability B is better, and how much could I lose?" — not just p."""
from __future__ import annotations

import numpy as np


def beta_binomial(control: np.ndarray, treat: np.ndarray, prior=(1.0, 1.0),
                  n_samples: int = 200_000, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    a0, b0 = prior
    ac, bc = a0 + control.sum(), b0 + len(control) - control.sum()
    at, bt = a0 + treat.sum(), b0 + len(treat) - treat.sum()

    sc = rng.beta(ac, bc, n_samples)
    st = rng.beta(at, bt, n_samples)
    p_treat_better = float((st > sc).mean())
    # expected loss = E[max(0, other - chosen)] for choosing each variant
    loss_choose_treat = float(np.maximum(sc - st, 0).mean())
    loss_choose_control = float(np.maximum(st - sc, 0).mean())
    return {
        "p_treat_better": round(p_treat_better, 4),
        "control_cr_mean": round(float(sc.mean()), 5),
        "treat_cr_mean": round(float(st.mean()), 5),
        "control_ci95": [round(float(np.percentile(sc, 2.5)), 5), round(float(np.percentile(sc, 97.5)), 5)],
        "treat_ci95": [round(float(np.percentile(st, 2.5)), 5), round(float(np.percentile(st, 97.5)), 5)],
        "expected_loss_choose_treat": round(loss_choose_treat, 6),
        "expected_loss_choose_control": round(loss_choose_control, 6),
        "decision": "ship treatment" if p_treat_better > 0.95 else
                    ("keep control" if p_treat_better < 0.05 else "inconclusive"),
    }
