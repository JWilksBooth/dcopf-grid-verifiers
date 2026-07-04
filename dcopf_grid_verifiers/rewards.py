"""Reward functions for the DC-OPF environment.

Design carried over from the v1 economic dispatch environment:
- Hard feasibility gate: infeasible dispatch => optimality reward is 0,
  regardless of how cheap it claims to be. This kills the dominant reward
  hack (quote an impossibly cheap dispatch that violates line limits).
- Graded optimality: reward decays with cost gap vs the LP optimum, so
  near-optimal congestion-aware dispatch scores partial credit.

Four rewards, weights chosen so feasibility dominates:
  1. format_reward      (0.10) - parseable JSON answer with correct arity
  2. feasibility_reward (0.30) - passes physics check (balance, bounds, limits)
  3. optimality_reward  (0.50) - gated on feasibility; exp decay in relative cost gap
  4. congestion_reward  (0.10) - gated on feasibility; credits correctly loading
                                 binding lines near (not over) their limits
"""

from __future__ import annotations

import json
import math
import re

import numpy as np

from .generator import Instance
from .solver import check_feasibility, solve_dcopf, S_BASE_MW


def parse_dispatch(completion: str) -> list[float] | None:
    """Extract {"dispatch_mw": [...]} from the completion (last valid match wins).

    Rejects non-finite values: json.loads accepts NaN/Infinity literals, and NaN
    in particular defeats comparison-based feasibility checks (every NaN
    comparison is False, so no violation is ever recorded). Without this guard,
    {"dispatch_mw": [NaN, ...]} would score full format+feasibility+optimality.
    """
    matches = re.findall(r'\{[^{}]*"dispatch_mw"[^{}]*\}', completion, re.DOTALL)
    for raw in reversed(matches):
        try:
            obj = json.loads(raw)
            vals = obj.get("dispatch_mw")
            if (isinstance(vals, list)
                    and all(_is_finite_number(v) for v in vals)):
                return [float(v) for v in vals]
        # ValueError covers JSONDecodeError AND the >4300-digit-int limit
        # ValueError json raises before our finite guard; RecursionError covers
        # deeply nested-bracket payloads. Both must score 0, not crash.
        except (ValueError, RecursionError, TypeError):
            continue
    return None


def _is_finite_number(v) -> bool:
    """True for finite int/float; False for bool, NaN, Inf, and integers too
    large for float (whose math.isfinite would raise OverflowError)."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return False
    try:
        return math.isfinite(float(v))
    except OverflowError:
        return False


def format_reward(completion: str, inst: Instance) -> float:
    d = parse_dispatch(completion)
    return 1.0 if (d is not None and len(d) == len(inst.generators)) else 0.0


def feasibility_reward(completion: str, inst: Instance) -> float:
    d = parse_dispatch(completion)
    if d is None:
        return 0.0
    ok, _ = check_feasibility(inst, d)
    return 1.0 if ok else 0.0


def optimality_reward(completion: str, inst: Instance,
                      optimal_cost: float | None = None) -> float:
    """exp(-5 * relative cost gap), hard-gated on feasibility.

    Tolerance-rent guard: check_feasibility allows +/-0.5 MW slack, so a
    dispatch that exploits it (under-serving load, or riding a bound) can cost
    LESS than the LP optimum; a clamped one-sided gap would score that 1.0. A
    feasible dispatch can never legitimately beat the optimum, so residual
    violations are priced at the most expensive unit's rate and any remaining
    cost below the optimum is penalized as strongly as cost above it.
    """
    d = parse_dispatch(completion)
    if d is None:
        return 0.0
    ok, _ = check_feasibility(inst, d)
    if not ok:
        return 0.0  # THE GATE — no optimality credit for physics violations
    if optimal_cost is None:
        sol = solve_dcopf(inst)
        if sol is None:
            return 0.0
        optimal_cost = sol["cost"]
    # Violations are settled at a penalty price ABOVE the highest offer (2x),
    # exactly like real imbalance settlement: pricing them at max_rate alone
    # would make shaving the priciest unit cost-neutral rather than losing.
    penalty_rate = 2.0 * max(g.cost_per_mwh for g in inst.generators)
    shortfall = max(0.0, sum(inst.loads_mw) - sum(d))
    bound_violation = sum(max(0.0, g.p_min_mw - p) + max(0.0, p - g.p_max_mw)
                          for g, p in zip(inst.generators, d))
    cost = sum(g.cost_per_mwh * p for g, p in zip(inst.generators, d))
    cost += (shortfall + bound_violation) * penalty_rate
    if optimal_cost <= 0:
        return 1.0 if cost <= 1e-6 else 0.0
    gap = abs(cost - optimal_cost) / optimal_cost  # symmetric: cheaper-than-optimal = abuse
    return math.exp(-5.0 * gap)


def congestion_reward(completion: str, inst: Instance) -> float:
    """Gated on feasibility. Credits solutions that load the network's binding
    lines to within 90-100% of their limits when the LP optimum also binds them.
    Rewards *understanding* congestion rather than over-conservative dispatch.
    """
    d = parse_dispatch(completion)
    if d is None:
        return 0.0
    ok, _ = check_feasibility(inst, d)
    if not ok:
        return 0.0
    sol = solve_dcopf(inst)
    if sol is None:
        return 0.0
    binding = [k for k, (f, l) in enumerate(zip(sol["flows_mw"], inst.lines))
               if abs(f) >= 0.98 * l.limit_mw]
    if not binding:
        return 1.0  # nothing to get right; don't penalize

    # flows implied by candidate dispatch
    n_bus = inst.n_buses
    inj = -np.array(inst.loads_mw, dtype=float)
    for g, p in zip(inst.generators, d):
        inj[g.bus] += p
    B = np.zeros((n_bus, n_bus))
    for l in inst.lines:
        b = S_BASE_MW / l.reactance
        B[l.from_bus, l.from_bus] += b
        B[l.to_bus, l.to_bus] += b
        B[l.from_bus, l.to_bus] -= b
        B[l.to_bus, l.from_bus] -= b
    theta = np.zeros(n_bus)
    theta[1:] = np.linalg.solve(B[1:, 1:], inj[1:])

    hits = 0
    for k in binding:
        l = inst.lines[k]
        flow = abs(S_BASE_MW / l.reactance * (theta[l.from_bus] - theta[l.to_bus]))
        if flow >= 0.90 * l.limit_mw:
            hits += 1
    return hits / len(binding)


REWARD_WEIGHTS = {
    "format_reward": 0.10,
    "feasibility_reward": 0.30,
    "optimality_reward": 0.50,
    "congestion_reward": 0.10,
}
