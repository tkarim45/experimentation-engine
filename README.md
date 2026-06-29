# experimentation-engine

An **A/B testing engine** that takes an experiment from *design* to *decision* with the four
things teams actually get wrong: undersized tests, p-value tunnel vision, throwing away free
power, and peeking at results until they look significant.

One command runs the whole analysis over a simulated experiment with a **known ground-truth
effect** — so every method is validated, not just asserted.

```bash
abtest                 # full report
abtest --json          # machine-readable
abtest --n 20000 --rel-lift 0.05 --seed 1
```

## What it does

| Module | Question it answers |
|---|---|
| `frequentist` | Two-proportion z-test, Welch t-test, **required sample size** & **achieved power** for an MDE — *"is the lift real, and was the test even powered to see it?"* |
| `bayesian` | Beta-Binomial posteriors → **P(treatment > control)** and **expected loss** of each decision — *"what's the probability B wins, and how much could shipping it cost me?"* |
| `cuped` | **CUPED** variance reduction using a pre-experiment covariate — same unbiased effect, tighter CI, more power for free |
| `sequential` | **mSPRT always-valid p-value** + a peeking simulation proving naive repeated peeking inflates false positives while always-valid does not |

## Measured results

From `abtest` on the default simulated experiment (true lift **+10%**, 5000/arm, seed 7):

```
DESIGN       true lift 10% | n/arm 5000 | need 12004/arm for 80% power | achieved power 0.44
FREQUENTIST  control 0.120 vs treat 0.140 | lift +16.3% | p=0.0036 | SIGNIFICANT
BAYESIAN     P(treat>control)=0.998 | E[loss|ship]≈0 | decision: SHIP TREATMENT
CUPED        variance reduction +51.3% | CI width 1.12 -> 0.78 | (free power)
PEEKING      naive repeated-peeking FP rate 21.3%  vs  always-valid 1.0%   (target ≤ 5%)
```

Three findings worth their own line:

- **Peeking is not a rounding error.** Checking a fixed-horizon test at 10 interim looks drives
  the false-positive rate to **21.3%** on null data — 4× the nominal 5%. The mSPRT always-valid
  p-value holds at **1.0%**. This is the single most common way teams ship nothing as something.
- **CUPED cut variance 51%** here (covariate ρ≈0.7). That CI shrink is equivalent to roughly
  *doubling* the sample size — at zero extra traffic cost.
- **The "significant" test was underpowered** (power 0.44 at the realized n). The engine flags
  this explicitly: a significant p-value from an underpowered test is a coin flip you got lucky on.

## Why it's built this way

Ground-truth simulation means the suite proves the methods *work*: a real lift is detected, a
true null is not (`test_ztest_null_not_significant`), Bayesian agrees with frequentist on strong
effects, CUPED measurably reduces variance, and naive peeking measurably inflates error. Nothing
here is asserted without a number behind it.

## Install & test

```bash
pip install -e ".[dev]"
pytest -q          # 8 passed
```

## Stack

Pure NumPy + SciPy — no black-box stats library. Two-proportion z, Welch t, power/sample-size,
Beta-Binomial Monte-Carlo posteriors, CUPED, and mSPRT all implemented from the formulas.

## License

MIT
