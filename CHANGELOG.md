# Changelog

Any entry marked **[dataset-changing]** alters the per-seed instance
distribution; downstream numbers (README validation table, baseline evals)
are regenerated in the same release.

## 0.2.2 — 2026-07-04

Second red-team pass (five-lens income-feasibility review, findings
independently verified):

- **[robustness] Uncaught parser crashes sealed.** A JSON integer over
  CPython's 4300-digit limit makes `json.loads` raise a bare `ValueError`
  (not `JSONDecodeError`), and deeply nested brackets raise `RecursionError`;
  both escaped the parser. `parse_dispatch` now catches `(ValueError,
  RecursionError, TypeError)`. Regression cases added to gate test 5. No score
  change for any valid completion; previously-crashing inputs now score 0.
- **[docs] Baseline reproduction fixed.** The Usage `vf-eval` command omitted
  `--max-tokens`; at the 4096 default, frontier rollouts truncate and the score
  collapses to ~0.5 — a reviewer would read the published 0.901 as inflated. The
  command now carries the flags and a warning sits under the Baseline table.
- **[hub] `rollouts_per_example` 3 -> 1** in `[tool.verifiers.eval]` so a
  one-click Hub eval matches the published single-rollout baselines at 1/3 the
  cost. `.prime/` gitignored; cross-repo roadmap links and premium figures
  reconciled to the shipped dataset.

## 0.2.1 — 2026-07-03

Red-team + expert-review hardening pass (adversarial multi-agent review;
findings independently verified before adoption):

- **[security] NaN/Infinity reward hack sealed.** `json.loads` accepts
  `NaN`/`Infinity` literals; NaN defeats comparison-based feasibility checks
  (every comparison is False), so `{"dispatch_mw": [NaN, ...]}` scored full
  format+feasibility+optimality. Non-finite values are now rejected in
  `parse_dispatch` and independently in `check_feasibility`. Gate test 5.
- **[security] Tolerance-rent exploit sealed.** Under-serving load within the
  0.5 MW feasibility tolerance produced a cost *below* the LP optimum, which a
  one-sided clamped gap scored 1.0. `optimality_reward` now prices residual
  violations at the most expensive unit's rate and penalizes below-optimum
  cost symmetrically. Gate test 6.
- **[robustness] OverflowError crash fixed:** integer literals too large for
  float no longer crash `parse_dispatch` (rejected like NaN/Inf).
- **[critical] verifiers >= 0.1.14 compatibility:** completion messages are
  Pydantic models, not dicts; the old `_text()` silently dropped them, zeroing
  all rewards under `vf-eval`. Now handles str / dict / Pydantic / content-parts.
- **[dataset-changing] Calibration adopted** (owner-ratified): tiered
  baseload/mid/peaker cost stack, `congestion_bias` 0.55 -> 0.45. Measured on
  the shipped dataset (seeds 0-299): 46% of instances defeat network-
  unconstrained merit-order dispatch; 9.2% median congestion premium
  (~44% / 11.6% at sweep time; see CALIBRATION.md).
- **[dataset-changing] Physical line-rating bounds:** loose ratings capped at
  1.0x total system load (a line rated above total load is impossible
  equipment), 15 MW floor on all ratings (keeps the 0.5 MW tolerance under
  ~3.3% of any rating). Measured distribution-neutral on the difficulty metric.
- **[dataset-changing] Retry-reseed collision fixed:** seed 0's rejection
  retries replayed seeds 1, 2, ... verbatim, duplicating dataset instances.
- Crosscheck test now fails if pandapower silently fails on >2% of instances
  (guards against a vacuous 0-mismatch pass).

## 0.2.0 — 2026-07-03

Initial DC-OPF environment: B-theta LP ground truth (scipy HiGHS),
pandapower `rundcopp` cross-validation, physics-gated rewards, calibration
harness (`calibration/measure.py`).
