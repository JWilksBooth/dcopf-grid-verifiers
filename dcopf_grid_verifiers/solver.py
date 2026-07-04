"""Ground-truth DC-OPF solver (B-theta LP formulation, scipy.optimize.linprog).

Decision variables: x = [P_g (n_gen), theta (n_bus)]
- theta_0 (slack) fixed to 0 via equality bounds
- Nodal balance:  sum_{g at bus b} P_g - sum_lines B_l * (theta_i - theta_j) incident = load_b
- Line limits:   |(theta_i - theta_j) / x_l| * S_base <= limit_mw
- Objective:     min sum_g cost_g * P_g

Independent of pandapower: this is the primary solver; pandapower is the
cross-check (see crosscheck.py). Two independent implementations agreeing on
objective value is the validation methodology.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from .generator import Instance

S_BASE_MW = 100.0
TOL = 1e-6


def solve_dcopf(inst: Instance) -> dict | None:
    """Solve DC-OPF. Returns dict with dispatch, flows, cost — or None if infeasible."""
    n_bus = inst.n_buses
    n_gen = len(inst.generators)
    n_line = len(inst.lines)
    n_var = n_gen + n_bus  # [P_g..., theta...]

    c = np.zeros(n_var)
    for i, g in enumerate(inst.generators):
        c[i] = g.cost_per_mwh

    # Equality: nodal balance, one row per bus
    A_eq = np.zeros((n_bus, n_var))
    b_eq = np.array(inst.loads_mw, dtype=float)
    for i, g in enumerate(inst.generators):
        A_eq[g.bus, i] = 1.0
    for l in inst.lines:
        b_coef = S_BASE_MW / l.reactance  # MW per rad
        # flow from->to = b_coef * (theta_from - theta_to); leaves from_bus, enters to_bus
        A_eq[l.from_bus, n_gen + l.from_bus] -= b_coef
        A_eq[l.from_bus, n_gen + l.to_bus] += b_coef
        A_eq[l.to_bus, n_gen + l.from_bus] += b_coef
        A_eq[l.to_bus, n_gen + l.to_bus] -= b_coef

    # Inequality: line limits, two rows per line (+/-)
    A_ub = np.zeros((2 * n_line, n_var))
    b_ub = np.zeros(2 * n_line)
    for k, l in enumerate(inst.lines):
        b_coef = S_BASE_MW / l.reactance
        A_ub[2 * k, n_gen + l.from_bus] = b_coef
        A_ub[2 * k, n_gen + l.to_bus] = -b_coef
        b_ub[2 * k] = l.limit_mw
        A_ub[2 * k + 1, n_gen + l.from_bus] = -b_coef
        A_ub[2 * k + 1, n_gen + l.to_bus] = b_coef
        b_ub[2 * k + 1] = l.limit_mw

    bounds = [(g.p_min_mw, g.p_max_mw) for g in inst.generators]
    bounds += [(0.0, 0.0)] + [(None, None)] * (n_bus - 1)  # theta_0 = 0

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=bounds, method="highs")
    if not res.success:
        return None

    dispatch = res.x[:n_gen]
    theta = res.x[n_gen:]
    flows = np.array([
        S_BASE_MW / l.reactance * (theta[l.from_bus] - theta[l.to_bus])
        for l in inst.lines
    ])
    return {
        "dispatch_mw": dispatch.tolist(),
        "theta_rad": theta.tolist(),
        "flows_mw": flows.tolist(),
        "cost": float(np.dot(c[:n_gen], dispatch)),
    }


def check_feasibility(inst: Instance, dispatch_mw: list[float],
                      tol_mw: float = 0.5) -> tuple[bool, list[str]]:
    """Physics-based feasibility check of a *proposed* dispatch.

    This is the anti-reward-hacking gate: a dispatch only earns optimality
    reward if it is physically feasible. Given a dispatch, bus angles are not
    free — with the slack fixed, injections determine angles uniquely on a
    connected network — so we solve for angles implied by the dispatch and
    verify balance and line limits.
    """
    n_bus = inst.n_buses
    n_gen = len(inst.generators)
    violations: list[str] = []

    if len(dispatch_mw) != n_gen:
        return False, [f"expected {n_gen} dispatch values, got {len(dispatch_mw)}"]

    # Defense in depth vs NaN/Inf: every violation test below is a comparison,
    # and every comparison against NaN is False — a NaN dispatch would record
    # zero violations and pass. Reject non-finite values outright.
    if not all(np.isfinite(p) for p in dispatch_mw):
        return False, ["non-finite dispatch value (NaN/Inf rejected)"]

    for i, (g, p) in enumerate(zip(inst.generators, dispatch_mw)):
        if p < g.p_min_mw - tol_mw or p > g.p_max_mw + tol_mw:
            violations.append(
                f"Gen {i} output {p:.1f} MW outside [{g.p_min_mw}, {g.p_max_mw}]")

    total_gen = sum(dispatch_mw)
    total_load = sum(inst.loads_mw)
    if abs(total_gen - total_load) > tol_mw:
        violations.append(
            f"power balance violated: gen {total_gen:.1f} MW vs load {total_load:.1f} MW")
        return False, violations  # angles undefined if unbalanced; stop here

    # Net injection per bus
    inj = -np.array(inst.loads_mw, dtype=float)
    for g, p in zip(inst.generators, dispatch_mw):
        inj[g.bus] += p

    # B matrix (susceptance Laplacian), reduced by slack bus 0
    B = np.zeros((n_bus, n_bus))
    for l in inst.lines:
        b = S_BASE_MW / l.reactance
        B[l.from_bus, l.from_bus] += b
        B[l.to_bus, l.to_bus] += b
        B[l.from_bus, l.to_bus] -= b
        B[l.to_bus, l.from_bus] -= b
    theta = np.zeros(n_bus)
    theta[1:] = np.linalg.solve(B[1:, 1:], inj[1:])

    for k, l in enumerate(inst.lines):
        flow = S_BASE_MW / l.reactance * (theta[l.from_bus] - theta[l.to_bus])
        if abs(flow) > l.limit_mw + tol_mw:
            violations.append(
                f"Line {k} ({l.from_bus}-{l.to_bus}) flow {flow:.1f} MW exceeds limit {l.limit_mw} MW")

    return len(violations) == 0, violations
