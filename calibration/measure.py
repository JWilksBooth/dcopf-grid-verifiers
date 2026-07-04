"""Calibration measurement + sweep harness for the DC-OPF generator.

Why this exists
---------------
"~50% of instances are congested" is only meaningful if congestion actually
changes the answer. This harness measures the metric that matters — the fraction
of instances where the *network-unconstrained merit-order dispatch is infeasible*
(it violates a line limit once you solve the DC power flow). Those are the
instances that genuinely defeat "just stack the cheapest units." It also reports
the cost premium congestion imposes and the generator's feasibility-rejection
rate, then sweeps the calibration knobs so a defensible configuration can be
chosen rather than inherited.

Run:
    python calibration/measure.py                 # full sweep + baseline
    python calibration/measure.py --n 500         # more instances per config
"""

import sys, os, json, argparse, random
from statistics import mean, median

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dcopf_grid_verifiers.generator import (
    generate_instance, build_instance, DEFAULT_GEN_CONFIG,
)
from dcopf_grid_verifiers.solver import solve_dcopf, S_BASE_MW

BIND_FRAC = 0.98   # a line is "binding" at >= 98% of its limit
TOL_MW = 0.5       # line-limit slack when calling a merit dispatch infeasible


def merit_order_dispatch(inst):
    """Network-free least-cost dispatch: commit every p_min, then buy the
    cheapest headroom until load is met. This is what merit-order-only reasoning
    (ignoring the network) produces."""
    load = sum(inst.loads_mw)
    disp = [g.p_min_mw for g in inst.generators]
    remaining = load - sum(disp)
    for i in sorted(range(len(inst.generators)),
                    key=lambda k: inst.generators[k].cost_per_mwh):
        if remaining <= 1e-9:
            break
        head = inst.generators[i].p_max_mw - disp[i]
        take = min(head, remaining)
        disp[i] += take
        remaining -= take
    return disp


def implied_flows(inst, dispatch):
    """MW flows implied by a dispatch via the slack-referenced B-theta solve."""
    n_bus = inst.n_buses
    inj = -np.array(inst.loads_mw, dtype=float)
    for g, p in zip(inst.generators, dispatch):
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
    return [S_BASE_MW / l.reactance * (theta[l.from_bus] - theta[l.to_bus])
            for l in inst.lines]


def measure_instance(inst):
    opt = solve_dcopf(inst)
    if opt is None:
        return None
    merit = merit_order_dispatch(inst)
    merit_cost = sum(g.cost_per_mwh * p for g, p in zip(inst.generators, merit))
    mflows = implied_flows(inst, merit)
    worst_merit_load = max(abs(f) / l.limit_mw for f, l in zip(mflows, inst.lines))
    merit_infeasible = any(abs(f) > l.limit_mw + TOL_MW
                           for f, l in zip(mflows, inst.lines))
    n_binding = sum(1 for f, l in zip(opt["flows_mw"], inst.lines)
                    if abs(f) >= BIND_FRAC * l.limit_mw)
    premium = (opt["cost"] - merit_cost) / merit_cost if merit_cost > 1e-9 else 0.0
    return {
        "merit_infeasible": merit_infeasible,
        "worst_merit_load": worst_merit_load,
        "binding": n_binding >= 1,
        "n_binding": n_binding,
        "premium": max(0.0, premium),
        "n_buses": inst.n_buses,
        "n_lines": len(inst.lines),
    }


def rejection_rate(cfg, k=500, seed_base=900_000):
    """Fraction of single raw draws that are infeasible (pre-resample). High
    rejection => the config biases the accepted sample toward easy instances."""
    fails = 0
    for i in range(k):
        rng = random.Random(seed_base + i)
        if solve_dcopf(build_instance(rng, **cfg)) is None:
            fails += 1
    return fails / k


def measure_config(cfg, n, seed_offset=0):
    rows = []
    for s in range(seed_offset, seed_offset + n):
        m = measure_instance(generate_instance(s, **cfg))
        if m:
            rows.append(m)
    infeas = [r for r in rows if r["merit_infeasible"]]
    prem_congested = [r["premium"] for r in infeas]
    return {
        "n": len(rows),
        "merit_infeasible_rate": len(infeas) / len(rows),
        "binding_rate": sum(r["binding"] for r in rows) / len(rows),
        "mean_premium_all": mean(r["premium"] for r in rows),
        "median_premium_congested": median(prem_congested) if prem_congested else 0.0,
        "max_premium": max(r["premium"] for r in rows),
        "mean_n_binding": mean(r["n_binding"] for r in rows),
        "mean_n_buses": mean(r["n_buses"] for r in rows),
        "mean_n_lines": mean(r["n_lines"] for r in rows),
        "rejection_rate": rejection_rate(cfg),
    }


def fmt_row(label, r):
    return (f"{label:<34} | {r['merit_infeasible_rate']*100:5.1f}% "
            f"| {r['binding_rate']*100:5.1f}% "
            f"| {r['median_premium_congested']*100:6.1f}% "
            f"| {r['max_premium']*100:6.1f}% "
            f"| {r['rejection_rate']*100:5.1f}% "
            f"| {r['mean_n_binding']:.2f} | {r['n']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=int(os.environ.get("N_INSTANCES", "300")))
    args = ap.parse_args()
    n = args.n

    header = (f"{'config':<34} | m-infeas | binding | med prem | max prem "
              f"| reject | binds | N")
    print("\nMetric legend:")
    print("  m-infeas = % instances where unconstrained merit-order dispatch violates a line limit  <-- THE difficulty metric")
    print("  binding  = % instances with >=1 line at >=98% of limit at the OPTIMUM (softer)")
    print("  med prem = median congestion cost premium over the m-infeasible subset")
    print("  reject   = % of raw draws rejected as infeasible (sample-bias indicator)")
    print(f"\n{header}\n{'-'*len(header)}")

    results = {}

    # Baseline = current shipped defaults
    base = measure_config({}, n)
    results["baseline_default"] = {"config": {}, "metrics": base}
    print(fmt_row("baseline (shipped default)", base))
    print(f"{'-'*len(header)}")

    # Sweep: congestion_bias x cost_model
    for cost_model in ("uniform", "stack"):
        for bias in (0.30, 0.45, 0.55, 0.70):
            cfg = {"congestion_bias": bias, "cost_model": cost_model}
            r = measure_config(cfg, n)
            key = f"bias={bias}_{cost_model}"
            results[key] = {"config": cfg, "metrics": r}
            print(fmt_row(f"bias={bias:<4} cost={cost_model}", r))
        print(f"{'-'*len(header)}")

    out = os.path.join(os.path.dirname(__file__), "sweep_results.json")
    with open(out, "w") as f:
        json.dump({"n_per_config": n, "default_config": DEFAULT_GEN_CONFIG,
                   "results": results}, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
