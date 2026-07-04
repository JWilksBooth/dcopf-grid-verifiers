# Reproducing the baselines

Calibration is *proven* only when the environment separates a weak model from a
strong one, with the frontier model **not** maxing it (headroom). Measured
result (in the README): claude-haiku-4-5 0.520 vs claude-opus-4-8 0.901 —
a wide gap with the frontier model at 90% feasibility, not saturated. Below is
how to reproduce it.

> **Windows note:** verifiers 0.1.14 crashes on Windows at import
> (`verifiers.v1` unconditionally imports the Unix-only `fcntl`). This machine
> carries a local patch (msvcrt fallback in
> `verifiers/envs/experimental/utils/file_locks.py` and `git_checkout_cache.py`)
> that is wiped if verifiers is reinstalled — re-apply it, run evals under WSL,
> or upstream the fix. The Hub itself runs Linux; this affects local eval only.

## Run it

```powershell
# 1. set your key (console.anthropic.com; the account needs a credit balance —
#    a $0-balance key authenticates but every rollout 400s)
$env:ANTHROPIC_API_KEY = "sk-ant-..."
$env:PYTHONIOENCODING = "utf-8"   # models emit θ₀/→ etc.; cp1252 console crashes the final print

cd dcopf-grid-verifiers

# 2. weak / cheap model (6k tokens is enough for Haiku-class)
vf-eval dcopf-grid-verifiers -p anthropic -m claude-haiku-4-5-20251001 -n 50 -r 1 `
  --max-tokens 6000 --save-results --disable-tui --disable-env-server

# 3. strong / frontier model — GIVE IT ROOM: at 6k max-tokens, half of Opus 4.8's
#    rollouts truncated mid-reasoning before emitting the JSON (scored 0 on format
#    while averaging 0.999 optimality on completed ones). 16k is a fair budget.
vf-eval dcopf-grid-verifiers -p anthropic -m claude-opus-4-8 -n 50 -r 1 `
  --max-tokens 16000 --save-results --disable-tui --disable-env-server
```

`--disable-env-server` is required on Windows (the ZMQ env-server subprocess
never becomes healthy there); harmless elsewhere. If the final console table
still crashes on encoding, the results are already saved — read
`outputs/evals/<env>--<model>/<run>/results.jsonl` directly.

Cross-vendor contrast (optional, via OpenRouter — often a wider, more listing-worthy gap):

```powershell
$env:OPENROUTER_API_KEY = "sk-or-..."
vf-eval dcopf-grid-verifiers -p openrouter -m openai/gpt-4o-mini -n 50 --save-results
vf-eval dcopf-grid-verifiers -p openrouter -m openai/gpt-4o      -n 50 --save-results
```

`--save-results` writes a JSON under `./outputs/` with per-reward breakdown
(format / feasibility / optimality / congestion) and the weighted total.

## How to read it

- **Total score gap wide, frontier < ~0.85** → well-calibrated. Ship it.
- **Cheap model already ~0.9** → too easy. Raise `congestion_bias`, or adopt the
  recommended `bias=0.45, cost=stack` (sharper premium ⇒ optimality reward
  punishes merit-order harder). Re-run.
- **Frontier ~0.3** → too hard or ambiguous. Check the `feasibility` sub-reward:
  if models fail *format/feasibility* rather than *optimality*, the prompt or the
  answer format is the problem, not the physics. Loosen or clarify the prompt.

Record the two totals + the per-reward split in the Hub listing and the bounty
application. That table is the differentiator.
