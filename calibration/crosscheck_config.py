"""Dual-solver cross-check under an arbitrary generator config.

The ground-truth guarantee (scipy HiGHS LP == pandapower DC-OPF) must survive any
recalibration. This runs that check for a chosen config so a proposed
calibration can be proven solver-valid before it is adopted as the default.

    python calibration/crosscheck_config.py --bias 0.45 --cost stack --n 150
"""

import sys, os, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dcopf_grid_verifiers.generator import generate_instance
from dcopf_grid_verifiers.solver import solve_dcopf
from dcopf_grid_verifiers.crosscheck import solve_with_pandapower
from measure import measure_instance  # same folder

REL_TOL, ABS_TOL = 1e-3, 1.0


def run(cfg, n):
    mism, pp_fail, worst, minf = 0, 0, 0.0, 0
    for s in range(n):
        inst = generate_instance(s, **cfg)
        lp = solve_dcopf(inst)
        m = measure_instance(inst)
        if m and m["merit_infeasible"]:
            minf += 1
        pp = solve_with_pandapower(inst)
        if pp is None:
            pp_fail += 1
            continue
        gap = abs(pp["cost"] - lp["cost"])
        rel = gap / max(abs(lp["cost"]), 1e-9)
        worst = max(worst, rel)
        if gap > ABS_TOL and rel > REL_TOL:
            mism += 1
            print(f"  MISMATCH seed {s}: LP ${lp['cost']:.2f} vs pp ${pp['cost']:.2f}")
    print(f"config={cfg or 'DEFAULT'}")
    print(f"  cross-check: {n} instances | mismatches={mism} | pp non-converged={pp_fail} "
          f"| worst rel gap={worst:.2e}")
    print(f"  merit-infeasible: {minf}/{n} ({100*minf/n:.0f}%)")
    assert mism == 0, "solver disagreement under this config — do not adopt"
    print("  DUAL-SOLVER OK under this config")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bias", type=float)
    ap.add_argument("--cost", choices=["uniform", "stack"])
    ap.add_argument("--n", type=int, default=150)
    a = ap.parse_args()
    cfg = {}
    if a.bias is not None:
        cfg["congestion_bias"] = a.bias
    if a.cost is not None:
        cfg["cost_model"] = a.cost
    run(cfg, a.n)
