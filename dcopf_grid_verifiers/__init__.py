"""dcopf-grid-verifiers: DC optimal power flow environment (verifiers spec).

Successor to the merit-order economic dispatch environment. Adds network
physics: bus angles, line reactances, and MW flow limits, so optimal dispatch
is no longer simple merit order — congestion forces out-of-merit dispatch,
which is exactly what generic annotators and naive LLMs get wrong.

Usage on Prime Intellect Environments Hub:
    prime env push  (from the project root; wheel built from pyproject.toml)

Local eval with verifiers:
    import verifiers as vf
    env = vf.load_environment("dcopf-grid-verifiers")
"""

from .generator import Instance, generate_instance, instance_to_prompt
from .solver import solve_dcopf, check_feasibility
from .rewards import (
    parse_dispatch, format_reward, feasibility_reward,
    optimality_reward, congestion_reward, REWARD_WEIGHTS,
)

__version__ = "0.2.3"

DEFAULT_NUM_EXAMPLES = 300


def build_dataset(num_examples: int = DEFAULT_NUM_EXAMPLES, seed_offset: int = 0):
    """Build (prompt, info) rows. info carries the instance + precomputed optimum."""
    rows = []
    for i in range(num_examples):
        inst = generate_instance(seed=seed_offset + i)
        sol = solve_dcopf(inst)
        rows.append({
            "question": instance_to_prompt(inst),
            "answer": str(round(sol["cost"], 2)),
            "info": {
                "instance": inst.to_dict(),
                "optimal_cost": sol["cost"],
                "optimal_dispatch": sol["dispatch_mw"],
            },
        })
    return rows


def load_environment(num_examples: int = DEFAULT_NUM_EXAMPLES,
                     seed_offset: int = 0, **kwargs):
    """verifiers entry point. Requires the `verifiers` package at runtime.

    seed_offset shifts the instance seed range, so trainers can build disjoint
    train/eval datasets: load_environment(1000) for training and
    load_environment(200, seed_offset=1000) for held-out evaluation. Instances
    are deterministic per seed; generation is fast enough for RL-scale datasets.
    """
    import verifiers as vf
    from datasets import Dataset

    rows = build_dataset(num_examples, seed_offset=seed_offset)
    dataset = Dataset.from_list(rows)

    def _inst(info) -> Instance:
        return Instance.from_dict(info["instance"])

    def fmt(completion, info, **kw):
        return format_reward(_text(completion), _inst(info))

    def feas(completion, info, **kw):
        return feasibility_reward(_text(completion), _inst(info))

    def opt(completion, info, **kw):
        return optimality_reward(_text(completion), _inst(info),
                                 optimal_cost=info["optimal_cost"])

    def cong(completion, info, **kw):
        return congestion_reward(_text(completion), _inst(info))

    def _text(completion) -> str:
        # Handles every completion shape verifiers may pass: raw string, list of
        # dict messages, or list of Pydantic message models (verifiers >= 0.1.14
        # passes Pydantic AssistantMessage objects, which are NOT dicts — an
        # isinstance(m, dict) filter silently drops them and zeroes all rewards).
        if isinstance(completion, str):
            return completion
        parts = []
        for m in completion:
            content = m.get("content") if isinstance(m, dict) else getattr(m, "content", None)
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):  # multimodal content parts
                for p in content:
                    t = p.get("text") if isinstance(p, dict) else getattr(p, "text", None)
                    if isinstance(t, str):
                        parts.append(t)
        return " ".join(parts)

    rubric = vf.Rubric(
        funcs=[fmt, feas, opt, cong],
        weights=[REWARD_WEIGHTS["format_reward"],
                 REWARD_WEIGHTS["feasibility_reward"],
                 REWARD_WEIGHTS["optimality_reward"],
                 REWARD_WEIGHTS["congestion_reward"]],
    )
    return vf.SingleTurnEnv(dataset=dataset, rubric=rubric, **kwargs)
