# Calibration

How the instance distribution was measured and chosen. This is the difference
between *inheriting* generator parameters and *authoring* them: every number in
`DEFAULT_GEN_CONFIG` is measured here against the metric that actually governs
task difficulty, and any change is proven to preserve the dual-solver ground
truth before adoption.

Reproduce: `python calibration/measure.py --n 300` (writes `sweep_results.json`).

## The metric that matters

"~50% of instances are congested" is only meaningful if congestion changes the
answer. A line can sit at its limit in the optimal solution while the naive
merit-order dispatch is *still feasible* — that instance does not punish bad
reasoning. So the headline metric here is **merit-infeasible rate**: the fraction
of instances where the network-unconstrained least-cost dispatch (commit every
`Pmin`, then buy cheapest headroom) **violates a line limit** once the DC power
flow is solved. Those are the instances that genuinely defeat merit-order-only
reasoning.

Secondary: **congestion cost premium** = `(optimal − merit_cost) / merit_cost`.
If this is ~0, the network is decorative. **Rejection rate** = fraction of raw
draws thrown out as infeasible; high rejection biases the accepted sample toward
easy instances.

## Sweep (n = 300 per config)

| config | merit-infeas | binding@opt | med premium (congested) | max premium | rejection |
|---|---|---|---|---|---|
| **baseline (bias 0.55, uniform)** | **46.3%** | 49.7% | 7.5% | 163% | 27.6% |
| bias 0.30, uniform | 28.3% | 31.0% | 3.5% | 64% | 14.6% |
| bias 0.45, uniform | 39.3% | 41.7% | 6.1% | 114% | 24.8% |
| bias 0.55, uniform | 46.3% | 49.7% | 7.5% | 163% | 27.6% |
| bias 0.70, uniform | 55.3% | 59.3% | 8.5% | 181% | 34.2% |
| bias 0.30, stack | 33.0% | 34.7% | 9.7% | 130% | 15.6% |
| **bias 0.45, stack (recommended)** | **43.7%** | 45.7% | **11.6%** | 156% | **21.0%** |
| bias 0.55, stack | 51.0% | 53.3% | 10.2% | 156% | 26.4% |
| bias 0.70, stack | 58.3% | 61.3% | 10.8% | 156% | 33.6% |

Reading it:
- **`binding@opt` overstates difficulty vs `merit-infeas`** (49.7% vs 46.3% at
  baseline). The README's "49% congested" is the soft metric; the real figure is ~46%.
- **`congestion_bias` trades difficulty against rejection.** Pushing bias to 0.70
  buys more merit-infeasible instances but rejects a third of all draws.
- **The realistic cost stack roughly doubles the median premium** (uniform 6–8%
  → stack 10–12%) at comparable difficulty, and does not worsen rejection.
  Congestion becomes economically sharp: an expensive *local* unit is forced on
  because a cheap *remote* unit cannot push its power across a constrained
  corridor — the actual LMP-separation phenomenon.

## Adopted configuration — ratified by the owner, 2026-07-03

**`congestion_bias = 0.45`, `cost_model = "stack"` is the shipped default**
(`DEFAULT_GEN_CONFIG` in `generator.py`), plus physical line-rating bounds
added in v0.2.1 (loose ratings capped at 1.0× total load, 15 MW floor —
measured distribution-neutral on the difficulty metric). Post-adoption
validation on the full default dataset (seeds 0–299): 0/300 dual-solver
mismatches (worst rel gap 6.9e-10, 1 pandapower non-convergence), 137/300
(46%) merit-infeasible, 142/300 (47%) binding at optimum, 9.2% median
congestion premium on the congested subset. Earlier configurations remain
reproducible via keyword overrides to `generate_instance`
(e.g. `congestion_bias=0.55, cost_model="uniform"`).

Rationale for the choice:
- 43.7% merit-infeasible — nearly half the instances genuinely require
  congestion-aware dispatch, measured on the sharp metric.
- 11.6% median premium (highest in the sweep) — congestion imposes a real,
  economically meaningful cost, so the optimal dispatch is worth getting right in
  dollars, not just in feasibility.
- 21.0% rejection — **lower** than the current default (27.6%), so *less* sample bias.
- Proven solver-valid: `python calibration/crosscheck_config.py --bias 0.45 --cost stack --n 150`
  → 0 mismatches, worst rel gap 4.4e-10.

Note: `calibration/measure.py`'s "baseline (shipped default)" row now measures
the adopted configuration; the sweep table above is the historical decision
record (its baseline row is the *pre*-adoption default, bias 0.55 / uniform).

## Grounding decisions — stylizations stated, not hidden

Design choices a power-systems practitioner will probe, with the honest answer
for each:

1. **Cost tiers** (`_COST_STACK` in `generator.py`): baseload $15–30, mid-merit
   $35–60, peaker $70–130 /MWh, mixed 40/40/20. Implied fuel basis: a
   **$5–8.5/MMBtu gas scenario** (deliberately stressed vs ~$3.5–4.5 Henry Hub
   spot in 2026) at heat rates ~7 MMBtu/MWh (CCGT mid-merit) and ~11–15
   (older frame CT / steam peakers) plus VOM. The high-fuel regime is chosen to
   widen tier separation — larger congestion premiums, sharper training
   signal. **Zero-marginal-cost renewables are deliberately omitted:** with $0
   bids, the interesting economics is curtailment, which needs its own reward
   design (candidate v0.3+ feature, behind a config flag through
   `calibration/measure.py` and the dual-solver check before adoption).
2. **Pmin drawn 0–15% of Pmax** models fully flexible committed capacity, not
   realistic thermal minimums (real CCGT ~40–50%, coal ~30–40%, frame CT ~30%).
   This is a **deliberate no-unit-commitment stylization**, and it is
   load-bearing: re-running the sweep with `pmin_frac_max=0.5` *drops* the
   merit-infeasible rate ~45% → ~37% and collapses the median congestion
   premium ~13% → ~7% (large committed Pmins spread injections and make naive
   merit dispatch feasible more often). Realistic tier-dependent Pmins are a
   design change that must be paired with re-tuned `congestion_bias` (~0.55–0.6)
   to preserve difficulty — deferred to the v0.3 UC-lite milestone.
3. **Line ratings** are drawn for congestion signal, not from conductor thermal
   data, but are bounded to physically possible values (enforced in the
   generator since v0.2.1): no rating above 1.0× total system load, none below
   15 MW. Ratings remain uncorrelated with line length/reactance — a known
   stylization.
4. **Reactances** `0.05–0.4 p.u.` on a 100 MVA base — plausible for a
   ~110–138 kV regime at ~15–150 km; real transmission spans wider (0.1–0.8).
5. **Why target ~45% congestion when real N-1-planned grids are congested far
   less often:** deliberately oversampling the congestion-relevant region so
   the environment has training/test signal; a realistic sample would be ~90%
   trivial merit-order and would not exercise the skill.

## Model discrimination — confirmed

Calibration is *proven* only when the environment separates a weak model from a
strong one. Measured (README Baseline table): **claude-haiku-4-5 0.520 vs
claude-opus-4-8 0.901** — a 0.38 gap, with the weak model at 52% feasibility
(it fails on physics, not formatting) and the frontier model at 90%, not
saturated. The frontier gap is partly reasoning budget: at 6k output tokens
half of Opus's rollouts truncate. Reproduce via `calibration/eval.md`.
