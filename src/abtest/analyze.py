"""CLI orchestrator — runs frequentist, Bayesian, CUPED, and a peeking demo over a simulated
experiment and prints a single decision-ready report."""
from __future__ import annotations

import argparse
import json

from . import bayesian, cuped, frequentist, sequential
from .data import simulate


def run(n_per_arm: int = 5000, base_rate: float = 0.12, rel_lift: float = 0.10,
        seed: int = 7) -> dict:
    exp = simulate(n_per_arm=n_per_arm, base_rate=base_rate, rel_lift=rel_lift, seed=seed)

    freq = frequentist.two_proportion_ztest(exp.control_conv, exp.treat_conv)
    n_needed = frequentist.required_sample_size(base_rate, rel_lift)
    power = round(frequentist.achieved_power(n_per_arm, base_rate, rel_lift), 4)
    bayes = bayesian.beta_binomial(exp.control_conv, exp.treat_conv)
    cup = cuped.apply(exp.control_metric, exp.control_pre, exp.treat_metric, exp.treat_pre)
    peek = sequential.peeking_demo(n_per_arm=min(n_per_arm, 2000), base_rate=base_rate)

    return {
        "design": {"true_rel_lift": rel_lift, "n_per_arm": n_per_arm,
                   "required_n_per_arm_for_mde": n_needed, "achieved_power": power},
        "frequentist_conversion": freq,
        "bayesian_conversion": bayes,
        "cuped_continuous_metric": cup,
        "peeking_false_positive_demo": peek,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run a full A/B experiment analysis.")
    ap.add_argument("--n", type=int, default=5000, help="users per arm")
    ap.add_argument("--base-rate", type=float, default=0.12)
    ap.add_argument("--rel-lift", type=float, default=0.10, help="true relative lift")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", action="store_true", help="emit raw JSON")
    args = ap.parse_args()

    report = run(args.n, args.base_rate, args.rel_lift, args.seed)
    if args.json:
        print(json.dumps(report, indent=2))
        return

    d, f, b, c, p = (report["design"], report["frequentist_conversion"],
                     report["bayesian_conversion"], report["cuped_continuous_metric"],
                     report["peeking_false_positive_demo"])
    print("=" * 64)
    print("  A/B EXPERIMENT ANALYSIS")
    print("=" * 64)
    print(f"\nDESIGN  true lift {d['true_rel_lift']:.0%} | n/arm {d['n_per_arm']} | "
          f"need {d['required_n_per_arm_for_mde']}/arm | power {d['achieved_power']:.2f}")
    print(f"\nFREQUENTIST  control {f['control_rate']:.3f} vs treat {f['treat_rate']:.3f} | "
          f"lift {f['rel_lift']:+.1%} | p={f['p_value']:.4f} | "
          f"{'SIGNIFICANT' if f['significant'] else 'n.s.'}")
    print(f"  95% CI abs diff: {f['ci95_abs_diff']}")
    print(f"\nBAYESIAN  P(treat>control)={b['p_treat_better']:.3f} | "
          f"E[loss|ship]={b['expected_loss_choose_treat']:.5f} | decision: {b['decision'].upper()}")
    print(f"\nCUPED  variance reduction {c['variance_reduction']:+.1%} | "
          f"CI width {c['ci_width_before']} -> {c['ci_width_after']} | "
          f"p {c['p_before']:.4f} -> {c['p_after']:.4f}")
    print(f"\nPEEKING  naive repeated peeking FP rate {p['naive_peeking_fp_rate']:.1%} "
          f"vs always-valid {p['always_valid_fp_rate']:.1%}  (target ≤ {p['alpha']:.0%})")
    print("=" * 64)


if __name__ == "__main__":
    main()
