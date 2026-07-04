"""Independent cross-validation of the LP solver using pandapower's DC OPF.

Same methodology as the v1 merit-order environment (merit-order solver vs
scipy.optimize.linprog): two independently implemented solvers must agree on
the optimal objective within tolerance across all generated instances.

pandapower models lines via loading percent of rated current; to impose exact
MW limits we model each corridor as a DC line equivalent using pandapower's
OPF with max_loading_percent on lines whose rating equals the MW limit.
Simplest faithful mapping: use pandapower networks built from the same
B-theta data via pp.create_line_from_parameters with x_ohm derived from the
p.u. reactance, and rated current chosen so 100% loading == limit_mw.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from .generator import Instance
from .solver import S_BASE_MW

VN_KV = 110.0  # arbitrary but consistent voltage level


def solve_with_pandapower(inst: Instance) -> dict | None:
    import pandapower as pp

    net = pp.create_empty_network(sn_mva=S_BASE_MW)
    buses = [pp.create_bus(net, vn_kv=VN_KV) for _ in range(inst.n_buses)]

    z_base = VN_KV ** 2 / S_BASE_MW  # ohm

    for l in inst.lines:
        x_ohm = l.reactance * z_base
        # max_i_ka such that limit_mw corresponds to 100% loading at vn_kv
        max_i_ka = l.limit_mw / (math.sqrt(3) * VN_KV)
        pp.create_line_from_parameters(
            net, buses[l.from_bus], buses[l.to_bus], length_km=1.0,
            r_ohm_per_km=0.0, x_ohm_per_km=x_ohm, c_nf_per_km=0.0,
            max_i_ka=max_i_ka, max_loading_percent=100.0,
        )

    for b, mw in enumerate(inst.loads_mw):
        if mw > 0:
            pp.create_load(net, buses[b], p_mw=mw, controllable=False)

    for i, g in enumerate(inst.generators):
        idx = pp.create_gen(
            net, buses[g.bus], p_mw=g.p_min_mw,
            min_p_mw=g.p_min_mw, max_p_mw=g.p_max_mw,
            controllable=True, slack=(i == 0),
        )
        pp.create_poly_cost(net, idx, "gen", cp1_eur_per_mw=g.cost_per_mwh)

    # ext grid not used; make gen 0's bus the slack via slack=True above
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pp.rundcopp(net, delta=1e-8)
    except Exception:
        return None
    if not net.OPF_converged:
        return None

    dispatch = net.res_gen["p_mw"].to_numpy()
    cost = float(sum(g.cost_per_mwh * p for g, p in zip(inst.generators, dispatch)))
    return {"dispatch_mw": dispatch.tolist(), "cost": cost}
