"""Validation harness.

1. Cross-check: scipy LP optimum vs pandapower DC-OPF optimum on N instances.
   Two independent solver implementations must agree on objective value.
2. Congestion audit: count instances where at least one line binds at the
   optimum (i.e. merit-order-only reasoning would be wrong).
3. Reward gate tests: feasible-optimal, feasible-suboptimal, infeasible-cheap
   (the reward hack), garbage output.
"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dcopf_grid_verifiers import (
    generate_instance, instance_to_prompt, solve_dcopf,
    format_reward, feasibility_reward, optimality_reward, congestion_reward,
)
from dcopf_grid_verifiers.crosscheck import solve_with_pandapower

N_INSTANCES = int(os.environ.get("N_INSTANCES", "200"))
REL_TOL = 1e-3   # 0.1% objective agreement
ABS_TOL = 1.0    # $1/h absolute floor for tiny objectives


def run_crosscheck():
    mismatches, binding_count, pp_fail = 0, 0, 0
    worst = 0.0
    for i in range(N_INSTANCES):
        inst = generate_instance(seed=i)
        lp = solve_dcopf(inst)
        assert lp is not None, f"seed {i}: generator produced infeasible instance"

        if any(abs(f) >= 0.98 * l.limit_mw for f, l in zip(lp["flows_mw"], inst.lines)):
            binding_count += 1

        pp_sol = solve_with_pandapower(inst)
        if pp_sol is None:
            pp_fail += 1
            continue
        gap = abs(pp_sol["cost"] - lp["cost"])
        rel = gap / max(abs(lp["cost"]), 1e-9)
        worst = max(worst, rel)
        if gap > ABS_TOL and rel > REL_TOL:
            mismatches += 1
            print(f"  MISMATCH seed {i}: LP ${lp['cost']:.2f} vs pandapower ${pp_sol['cost']:.2f}")

    print(f"cross-check: {N_INSTANCES} instances | mismatches={mismatches} | "
          f"pandapower non-converged={pp_fail} | worst rel gap={worst:.2e}")
    print(f"congestion:  {binding_count}/{N_INSTANCES} instances have a binding line at optimum "
          f"({100*binding_count/N_INSTANCES:.0f}%) — these defeat pure merit-order reasoning")
    assert mismatches == 0, "solver disagreement — do not ship"
    # guard against a vacuous pass: if pandapower silently failed on many
    # instances, mismatches==0 would mean nothing
    assert pp_fail <= 0.02 * N_INSTANCES, (
        f"pandapower failed on {pp_fail}/{N_INSTANCES} instances — crosscheck is not covering the sample")
    assert binding_count >= 0.2 * N_INSTANCES, "too few congested instances — raise congestion_bias"
    return mismatches, binding_count, pp_fail


def run_reward_gate_tests():
    inst = None
    # find a congested instance so suboptimal-but-feasible differs from optimal
    for s in range(500):
        cand = generate_instance(seed=s)
        sol = solve_dcopf(cand)
        if any(abs(f) >= 0.98 * l.limit_mw for f, l in zip(sol["flows_mw"], cand.lines)):
            inst, opt = cand, sol
            break
    assert inst is not None

    def answer(d): return json.dumps({"dispatch_mw": [round(x, 3) for x in d]})

    # 1. optimal answer
    a = answer(opt["dispatch_mw"])
    assert format_reward(a, inst) == 1.0
    assert feasibility_reward(a, inst) == 1.0
    assert optimality_reward(a, inst, opt["cost"]) > 0.99
    print("gate test 1 (optimal answer): PASS")

    # 2. infeasible-cheap reward hack: everything from the cheapest generator
    cheap = min(range(len(inst.generators)), key=lambda i: inst.generators[i].cost_per_mwh)
    hack = [0.0] * len(inst.generators)
    hack[cheap] = sum(inst.loads_mw)
    a = answer(hack)
    r_opt = optimality_reward(a, inst, opt["cost"])
    assert r_opt == 0.0, f"reward hack scored {r_opt} — gate broken"
    print("gate test 2 (infeasible-cheap hack): PASS — optimality reward 0.0")

    # 3. garbage output
    assert format_reward("the answer is forty-two", inst) == 0.0
    assert optimality_reward("the answer is forty-two", inst, opt["cost"]) == 0.0
    print("gate test 3 (garbage output): PASS")

    # 4. feasible but suboptimal: solve with *inverted* costs — same feasible
    # set, so the result is guaranteed feasible for the true instance, but
    # dispatches expensive units first (strictly worse objective when costs differ)
    from copy import deepcopy
    inst2 = deepcopy(inst)
    max_c = max(g.cost_per_mwh for g in inst2.generators)
    for g in inst2.generators:
        g.cost_per_mwh = max_c + 1.0 - g.cost_per_mwh
    sub = solve_dcopf(inst2)
    assert sub is not None, "same feasible set — must solve"
    a = answer(sub["dispatch_mw"])
    assert feasibility_reward(a, inst) == 1.0
    r = optimality_reward(a, inst, opt["cost"])
    assert 0.0 < r < 1.0, f"expected partial credit, got {r}"
    print(f"gate test 4 (feasible suboptimal): PASS — partial credit {r:.3f}")

    # 5. non-finite attack: json accepts NaN/Infinity literals, and every
    # comparison against NaN is False, so an unguarded feasibility check
    # records zero violations. Must score 0 on every reward.
    n_gen = len(inst.generators)
    for bad in ("NaN", "Infinity", "-Infinity"):
        a = '{"dispatch_mw": [' + ", ".join([bad] * n_gen) + ']}'
        assert format_reward(a, inst) == 0.0, f"{bad}: format gate broken"
        assert feasibility_reward(a, inst) == 0.0, f"{bad}: feasibility gate broken"
        assert optimality_reward(a, inst, opt["cost"]) == 0.0, f"{bad}: optimality gate broken"
        assert congestion_reward(a, inst) == 0.0, f"{bad}: congestion gate broken"
    print("gate test 5 (NaN/Infinity attack): PASS — all rewards 0.0")

    # 6. tolerance-rent attack: shave MW off an expensive dispatched unit so
    # the answer stays inside the 0.5 MW feasibility tolerance but costs LESS
    # than the LP optimum. A one-sided clamped gap would score this 1.0.
    shaved = list(opt["dispatch_mw"])
    victim = max((i for i in range(len(shaved))
                  if shaved[i] >= inst.generators[i].p_min_mw + 0.45),
                 key=lambda i: inst.generators[i].cost_per_mwh, default=None)
    assert victim is not None, "no unit dispatched above its floor — pick another seed"
    shaved[victim] -= 0.4
    a = answer(shaved)
    assert feasibility_reward(a, inst) == 1.0  # within tolerance by design
    r_honest = optimality_reward(answer(opt["dispatch_mw"]), inst, opt["cost"])
    r_shaved = optimality_reward(a, inst, opt["cost"])
    assert r_shaved < 0.999 and r_shaved < r_honest, (
        f"tolerance-rent attack scored {r_shaved} vs honest {r_honest} — guard broken")
    print(f"gate test 6 (tolerance-rent attack): PASS — {r_shaved:.4f} < honest {r_honest:.4f}")


if __name__ == "__main__":
    run_reward_gate_tests()
    run_crosscheck()
    print("ALL VALIDATION PASSED")
