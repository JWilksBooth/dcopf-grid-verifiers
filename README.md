# dcopf-grid-verifiers

DC Optimal Power Flow (DC-OPF) RL environment with transmission line limits. Successor to a merit-order economic dispatch environment: this version adds network physics (bus angles, line reactances, MW flow limits), so pure merit-order reasoning fails on a measured **46% of instances** — the network-unconstrained least-cost dispatch actually violates a line limit, and congestion forces out-of-merit dispatch. That is exactly what naive LLM reasoning gets wrong.

## Why this environment is hard to reward-hack

The dominant hack in dispatch tasks is quoting an impossibly cheap dispatch that violates physics. Here, the optimality reward is **hard-gated on a physics feasibility check**: given a proposed dispatch, bus angles are uniquely determined (slack-referenced B-theta solve), so power balance, generator bounds, and every line flow are verified before any optimality credit is granted. An infeasible answer scores 0 on optimality no matter how cheap it claims to be.

The gate is red-teamed in `tests/test_validation.py` (6 attack tests): the infeasible-cheap dispatch, garbage output, the subtler `NaN`/`Infinity` JSON attack (`json.loads` accepts these literals, and NaN defeats any comparison-based check because every comparison against NaN is `False`), and the tolerance-rent attack (under-serving load inside the ±0.5 MW feasibility tolerance to come in *below* the LP optimum — residual violations are settled at a 2× penalty price, real imbalance-market style, and below-optimum cost is penalized symmetrically). Non-finite values are rejected at parse *and* inside the physics check itself.

## Ground truth validation

Two independent implementations of the same DC-OPF formulation must agree (cross-implementation validation — both share DC power-flow physics; this catches modeling and coding errors, not model-form error):

- **Primary:** B-theta LP formulation solved with `scipy.optimize.linprog` (HiGHS)
- **Cross-check:** `pandapower.rundcopp` on the same networks (separately implemented modeling path: ohmic line parameters, current-based ratings)

Validation run (300 randomly generated instances, seeds 0–299 — the full default dataset):

| Metric | Result |
|---|---|
| Objective mismatches (>0.1% rel and >$1/h abs) | **0 / 300** |
| Worst relative objective gap | 6.9e-10 |
| pandapower non-convergence | 1 / 300 (test fails if >2%) |
| Merit-order dispatch infeasible (violates a line limit) | **137 / 300 (46%)** |
| Instances with a binding line at optimum | 142 / 300 (47%) |
| Median congestion cost premium (congested subset) | 9.2% |

The distribution behind these numbers is measured and tuned, not inherited — see [CALIBRATION.md](CALIBRATION.md) for the metric definitions, the parameter sweep, and the rationale for the chosen configuration.

## Baseline results

50 instances (seeds 0–49), 1 rollout each, default sampling, July 2026:

| Model | total | format | feasibility | optimality | congestion |
|---|---|---|---|---|---|
| claude-haiku-4-5 (6k tokens) | **0.520** | 1.000 | 0.520 | 0.432 | 0.480 |
| claude-opus-4-8 (16k tokens) | **0.901** | 0.920 | 0.900 | 0.898 | 0.900 |

> **Reproducing this: pass `--max-tokens`.** Anthropic endpoints default to 4096 output tokens, at which frontier rollouts truncate mid-derivation and the score collapses to ~0.5 (measured). Use `--max-tokens 16000` for Opus-class, `6000` for Haiku-class, with `-r 1`.

Reading: the weak model formats perfectly but only 52% of its dispatches survive
the physics check — it loses on feasibility, not parsing. The frontier model
reaches 90% feasibility and, *when feasible, averages 0.998 optimality* — its
remaining gap is reasoning budget (4/50 rollouts truncated at 16k tokens) and
occasional congestion misreads. Frontier reasoning-token demand is itself part
of the task's difficulty: at 6k tokens, half of the frontier model's rollouts
truncate before emitting an answer.

Reproduce:

```bash
vf-eval dcopf-grid-verifiers -p anthropic -m claude-haiku-4-5-20251001 -n 50 -r 1 --max-tokens 6000 --save-results
vf-eval dcopf-grid-verifiers -p anthropic -m claude-opus-4-8 -n 50 -r 1 --max-tokens 16000 --save-results
```

## Example failure (claude-haiku-4-5, from the baseline run)

> *"...I need to solve the AC power flow equations. After careful analysis of the feasible region, the optimal dispatch appears to be: P0 = 15.2 MW, P1 = 90.3 MW, P2 = 86.7 MW. This satisfies: Total: 192.2 OK, Bounds: All within limits OK"*

The model verifies generation balance and generator bounds, then declares victory — but never solves the network. Its dispatch drives **Line 0 to -44.4 MW against a 39.6 MW limit** (and it mistakes this DC problem for AC power flow). Feasibility reward: **0**. Confident, well-formatted, and physically impossible — exactly the failure mode the environment exists to catch.

## Rewards (weighted rubric)

| Reward | Weight | Description |
|---|---|---|
| `format_reward` | 0.10 | Parseable `{"dispatch_mw": [...]}` with correct arity, finite values only |
| `feasibility_reward` | 0.30 | Passes physics check: balance, gen bounds, all line limits |
| `optimality_reward` | 0.50 | Gated on feasibility; `exp(-5 × relative cost gap)` vs LP optimum |
| `congestion_reward` | 0.10 | Gated on feasibility; credits correctly loading binding lines to 90–100% of limit |

## Usage

```bash
# local install
pip install -e ".[crosscheck]"

# run validation harness (solver cross-check + reward gate tests)
N_INSTANCES=200 python tests/test_validation.py

# measure the instance-difficulty distribution / re-run the calibration sweep
python calibration/measure.py --n 300

# evaluate a model via verifiers CLI (give frontier models room to reason —
# see the warning under the Baseline table; -r 1 to match the published numbers)
vf-eval dcopf-grid-verifiers -p anthropic -m <model> -n 50 -r 1 --max-tokens 16000
```

```python
import verifiers as vf
env = vf.load_environment("dcopf-grid-verifiers", num_examples=300)
```

## Instance generation

Random connected networks (spanning tree + loop edges, 4–8 buses), 2–4 generators, loads on ~2/3 of buses. Generator costs are drawn from a **tiered merit stack** (baseload / mid-merit / peaker) rather than a flat distribution, so congestion is economically sharp: when a constrained corridor blocks a cheap remote unit, an expensive *local* unit must run — the mechanism behind real-world locational price separation. Units are modeled as fully flexible committed capacity (low Pmin — a deliberate no-unit-commitment stylization, documented with its measured difficulty trade-off in CALIBRATION.md). Line limits are drawn from a bimodal (tight/loose) distribution bounded to physically possible values (no rating above total system load, 15 MW floor) and calibrated so ~46% of instances defeat network-unconstrained merit-order dispatch (measured, see [CALIBRATION.md](CALIBRATION.md)). Every generated instance is solved at generation time; infeasible draws are rejected. Fully deterministic per seed.

The congestion rate is deliberately oversampled relative to a real N-1-planned grid, where binding congestion is far rarer: a realistic sample would be ~90% trivial merit-order instances and would not exercise the skill being tested.

## The vertical

- [economic-dispatch](https://github.com/JWilksBooth/economic-dispatch) — merit-order control rung (frontier-saturated by design).
- **dcopf-grid-verifiers** (this) — congestion / space.
- [multiperiod-dispatch](https://github.com/JWilksBooth/multiperiod-dispatch) — ramp coupling / time; shipped, 74% defeat per-period merit order.
- [n1-contingency-dispatch](https://github.com/JWilksBooth/n1-contingency-dispatch) — N-1 security / uncertainty; shipped, 68% defeat the plain OPF optimum.
- [nodal-pricing-lmp](https://github.com/JWilksBooth/nodal-pricing-lmp) — LMP price-formation / dual space; shipped, 44% out-of-band.

Robotics vertical (same validation discipline): [robot-cycle-time](https://github.com/JWilksBooth/robot-cycle-time) (motion), [bimanual-chore-scheduling](https://github.com/JWilksBooth/bimanual-chore-scheduling) (orchestration).
