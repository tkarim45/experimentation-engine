import numpy as np

from abtest import bayesian, cuped, frequentist, sequential
from abtest.analyze import run
from abtest.data import simulate


def test_simulate_shapes_and_truth():
    exp = simulate(n_per_arm=1000, rel_lift=0.10)
    assert exp.control_conv.shape == (1000,)
    assert exp.true_lift == 0.10
    assert exp.treat_conv.mean() > exp.control_conv.mean()  # real effect present


def test_ztest_detects_real_lift():
    exp = simulate(n_per_arm=20000, base_rate=0.12, rel_lift=0.15, seed=1)
    res = frequentist.two_proportion_ztest(exp.control_conv, exp.treat_conv)
    assert res["significant"] and res["rel_lift"] > 0


def test_ztest_null_not_significant():
    rng = np.random.default_rng(3)
    c = rng.binomial(1, 0.12, 5000)
    t = rng.binomial(1, 0.12, 5000)
    assert not frequentist.two_proportion_ztest(c, t)["significant"]


def test_sample_size_and_power_monotonic():
    n_small = frequentist.required_sample_size(0.12, 0.20)
    n_big = frequentist.required_sample_size(0.12, 0.05)   # smaller MDE -> more samples
    assert n_big > n_small
    assert 0 < frequentist.achieved_power(5000, 0.12, 0.10) <= 1


def test_bayesian_agrees_with_strong_effect():
    exp = simulate(n_per_arm=20000, base_rate=0.12, rel_lift=0.20, seed=2)
    res = bayesian.beta_binomial(exp.control_conv, exp.treat_conv)
    assert res["p_treat_better"] > 0.95
    assert res["expected_loss_choose_treat"] < res["expected_loss_choose_control"]


def test_cuped_reduces_variance():
    exp = simulate(n_per_arm=8000, seed=4)
    res = cuped.apply(exp.control_metric, exp.control_pre, exp.treat_metric, exp.treat_pre)
    assert res["variance_reduction"] > 0.2           # ρ≈0.7 -> sizable reduction
    assert res["ci_width_after"] < res["ci_width_before"]


def test_peeking_inflates_naive_not_always_valid():
    res = sequential.peeking_demo(n_per_arm=1500, n_sims=200, seed=5)
    # naive repeated peeking blows past alpha; always-valid stays controlled
    assert res["naive_peeking_fp_rate"] > res["alpha"]
    assert res["always_valid_fp_rate"] <= res["naive_peeking_fp_rate"]


def test_run_report_keys():
    rep = run(n_per_arm=3000)
    for k in ("design", "frequentist_conversion", "bayesian_conversion",
              "cuped_continuous_metric", "peeking_false_positive_demo"):
        assert k in rep
