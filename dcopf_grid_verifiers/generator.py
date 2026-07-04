"""Random DC-OPF instance generator.

Generates connected power networks with:
- N buses (bus 0 = slack)
- Lines with reactances and MW flow limits (spanning tree + extra loops)
- Generators with linear costs, min/max MW
- Loads (MW)

Guarantees feasibility by construction-check: every instance is solved by the
LP solver at generation time; infeasible draws are rejected and resampled.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict


@dataclass
class Line:
    from_bus: int
    to_bus: int
    reactance: float   # p.u.
    limit_mw: float    # thermal limit, MW (absolute value of flow)


@dataclass
class Generator:
    bus: int
    cost_per_mwh: float
    p_min_mw: float
    p_max_mw: float


@dataclass
class Instance:
    n_buses: int
    lines: list[Line] = field(default_factory=list)
    generators: list[Generator] = field(default_factory=list)
    loads_mw: list[float] = field(default_factory=list)  # length n_buses

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Instance":
        return Instance(
            n_buses=d["n_buses"],
            lines=[Line(**l) for l in d["lines"]],
            generators=[Generator(**g) for g in d["generators"]],
            loads_mw=list(d["loads_mw"]),
        )


def _random_connected_topology(rng: random.Random, n: int, extra_edge_prob: float) -> list[tuple[int, int]]:
    """Random spanning tree + extra edges -> connected graph with loops.

    Loops are what make line limits bind non-trivially (parallel paths share flow
    per susceptance, so a cheap generator can be curtailed by a congested line).
    """
    edges: set[tuple[int, int]] = set()
    nodes = list(range(n))
    rng.shuffle(nodes)
    # spanning tree over shuffled order
    for i in range(1, n):
        a, b = nodes[i], nodes[rng.randrange(i)]
        edges.add((min(a, b), max(a, b)))
    # extra edges to create loops
    for a in range(n):
        for b in range(a + 1, n):
            if (a, b) not in edges and rng.random() < extra_edge_prob:
                edges.add((a, b))
    return sorted(edges)


# --- Calibration knobs -------------------------------------------------------
# Every generation constant is a named default here so the instance distribution
# can be measured and tuned (see calibration/measure.py and CALIBRATION.md).
# Current defaults are the ADOPTED calibration (sweep of 2026-07, ratified by
# the owner): congestion_bias=0.45 + tiered cost stack + physical line-rating
# bounds. Measured on the sharp metric: ~44% of instances make the
# network-unconstrained merit-order dispatch infeasible; median congestion cost
# premium ~12%; draw rejection ~21%. Any change to these values changes the
# generated dataset and requires re-running the dual-solver validation
# (tests/test_validation.py) and refreshing the README/CALIBRATION.md tables.
DEFAULT_GEN_CONFIG: dict = {
    "n_buses_range": (4, 8),
    "extra_edge_prob": 0.25,
    "reactance_range": (0.05, 0.4),   # p.u. on 100 MVA base
    "load_range": (20.0, 120.0),      # MW per load bus
    "n_gen_range": (2, 4),
    "cap_range": (0.5, 1.5),          # relative generator size weights
    "cap_scale_range": (1.3, 2.0),    # aggregate capacity as multiple of load
    "pmin_frac_max": 0.15,            # p_min drawn in [0, frac] * p_max
    "cost_model": "stack",            # "uniform" | "stack"
    "cost_range": (10.0, 80.0),       # $/MWh, used when cost_model == "uniform"
    "congestion_bias": 0.45,          # P(a line gets the tight limit band)
    "tight_frac": (0.10, 0.35),       # tight limit as fraction of total load
    "loose_frac": (0.6, 1.0),         # loose limit as fraction of total load
    "limit_floor_mw": 15.0,           # no line rated below this (transmission, not a feeder)
}
# loose_frac capped at 1.0x total load: a line rated above total system load is
# impossible equipment at any voltage class and reads as fake data to a power
# engineer (ratings appear verbatim in every prompt). Measured on the sweep
# harness: (0.6, 1.0) vs the old (0.8, 1.5) is distribution-neutral
# (merit-infeasible 44.0% vs 44.7%, rejection 21.2% vs 21.0%). The 15 MW floor
# keeps the 0.5 MW feasibility tolerance under ~3.3% of any line's rating.

# Realistic merit-order stack (used when cost_model == "stack"): baseload /
# mid-merit / peaker tiers, so congestion forces an *expensive local* unit to
# run when a cheap remote unit cannot get its power across a constrained
# corridor. Tiers are proposed defaults to be validated against real fuel-cost
# data, not authoritative.
_COST_STACK = ((0.40, 15.0, 30.0), (0.40, 35.0, 60.0), (0.20, 70.0, 130.0))


def _draw_cost(rng: random.Random, cost_model: str, cost_range: tuple[float, float]) -> float:
    if cost_model == "uniform":
        return round(rng.uniform(*cost_range), 2)
    if cost_model == "stack":
        u = rng.random()
        acc = 0.0
        for prob, lo, hi in _COST_STACK:
            acc += prob
            if u < acc:
                return round(rng.uniform(lo, hi), 2)
        return round(rng.uniform(*_COST_STACK[-1][1:]), 2)
    raise ValueError(f"unknown cost_model {cost_model!r}")


def build_instance(rng: random.Random, **cfg) -> Instance:
    """Construct ONE instance from an rng (no feasibility check, no retry).

    Exposed so the calibration harness can measure the raw (pre-rejection) draw
    distribution. `generate_instance` wraps this with a feasibility-reject loop.
    """
    p = {**DEFAULT_GEN_CONFIG, **cfg}
    n = rng.randint(*p["n_buses_range"])
    topo = _random_connected_topology(rng, n, extra_edge_prob=p["extra_edge_prob"])

    lines = []
    for (a, b) in topo:
        x = rng.uniform(*p["reactance_range"])
        lines.append(Line(a, b, round(x, 4), 0.0))

    # loads on ~2/3 of buses
    loads = [0.0] * n
    load_buses = rng.sample(range(n), k=max(2, (2 * n) // 3))
    for bus in load_buses:
        loads[bus] = round(rng.uniform(*p["load_range"]), 1)
    total_load = sum(loads)

    n_gen = rng.randint(p["n_gen_range"][0], min(p["n_gen_range"][1], n))
    gen_buses = rng.sample(range(n), k=n_gen)
    caps = [rng.uniform(*p["cap_range"]) for _ in range(n_gen)]
    scale = total_load * rng.uniform(*p["cap_scale_range"]) / sum(caps)
    gens = []
    for bus, c in zip(gen_buses, caps):
        p_max = round(c * scale, 1)
        p_min = round(rng.uniform(0.0, p["pmin_frac_max"]) * p_max, 1)
        cost = _draw_cost(rng, p["cost_model"], p["cost_range"])
        gens.append(Generator(bus, cost, p_min, p_max))

    # line limits: tight (may bind) vs loose, floored so no rating is small
    # enough that the 0.5 MW feasibility tolerance becomes material slack
    floor = p["limit_floor_mw"]
    for ln in lines:
        if rng.random() < p["congestion_bias"]:
            ln.limit_mw = round(max(floor, rng.uniform(*p["tight_frac"]) * total_load), 1)
        else:
            ln.limit_mw = round(max(floor, rng.uniform(*p["loose_frac"]) * total_load), 1)

    return Instance(n, lines, gens, loads)


def generate_instance(seed: int, max_attempts: int = 200, **cfg) -> Instance:
    """Generate one feasible DC-OPF instance (deterministic per seed).

    Accepts any key in DEFAULT_GEN_CONFIG as a keyword override (e.g.
    congestion_bias=0.7, cost_model="uniform"). Infeasible draws are rejected
    and resampled from a reseeded stream. The reseed offsets by (seed + 1) so
    retry streams can never collide with another seed's primary stream (with
    the old `seed * K + attempt + 1` form, seed 0's retries replayed seeds
    1, 2, ... verbatim, duplicating instances in the dataset).
    """
    from .solver import solve_dcopf  # local import to avoid cycle at module load

    rng = random.Random(seed)
    for attempt in range(max_attempts):
        inst = build_instance(rng, **cfg)
        if solve_dcopf(inst) is not None:
            return inst
        rng = random.Random((seed + 1) * 1_000_003 + attempt)
    raise RuntimeError(f"Could not generate feasible instance for seed {seed}")


def instance_to_prompt(inst: Instance) -> str:
    """Render an instance as a natural-language dispatch task for an LLM."""
    lines_txt = "\n".join(
        f"  Line {i}: bus {l.from_bus} <-> bus {l.to_bus}, reactance {l.reactance} p.u., "
        f"flow limit {l.limit_mw} MW"
        for i, l in enumerate(inst.lines)
    )
    gens_txt = "\n".join(
        f"  Gen {i}: at bus {g.bus}, cost ${g.cost_per_mwh}/MWh, "
        f"output range [{g.p_min_mw}, {g.p_max_mw}] MW"
        for i, g in enumerate(inst.generators)
    )
    loads_txt = "\n".join(
        f"  Bus {b}: {mw} MW" for b, mw in enumerate(inst.loads_mw) if mw > 0
    )
    return f"""You are a power system operator solving a DC optimal power flow problem.

Network: {inst.n_buses} buses (bus 0 is the slack/reference bus).

Transmission lines (DC power flow; bus angles theta in radians; the MW flow on a line from bus i to bus j is: flow_MW = 100 * (theta_i - theta_j) / reactance_pu, i.e. per-unit flow on a 100 MVA base converted to MW; the limit applies to |flow_MW| in both directions):
{lines_txt}

Generators:
{gens_txt}

Loads:
{loads_txt}

Find the generator dispatch that minimizes total cost ($/h) while:
1. Total generation equals total load ({sum(inst.loads_mw):.1f} MW)
2. Each generator stays within its output range
3. No line flow exceeds its MW limit (DC power flow physics with the given reactances)

Report each MW value to at least one decimal place. Constraints are verified
with a +/-0.5 MW tolerance, but imbalance or bound violations inside that
tolerance are penalized in the cost scoring, so target exact feasibility.

Respond with your final answer as JSON on the last line, exactly in this format:
{{"dispatch_mw": [<Gen 0 MW>, <Gen 1 MW>, ...]}}"""
