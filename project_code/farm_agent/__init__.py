"""Modular agentic baseline for the farm simulation.

The harness used to be one 2,000-line script; it is now this package plus a thin
entry script, so the documented invocation is unchanged::

    python agentic_approach_baseline_farm_sim.py --model qwen3:8b --preset easy --seed 1
    python agentic_approach_baseline_farm_sim.py --check-api
    python -m farm_agent --check-api            # same entry point

Module map (data flows the same way it did in the single file):

    runtime      mutable run state - the old module-level globals live here
    config       config.yaml, argparse, log file setup
    api          the HTTP layer (GET /api/v1/state, POST /api/v1/action)
    tracking     visited tiles, jumps, revisits and the per-episode reset
    budget       ticks/minutes, the `budget:` line and the `advance` clamp
    rendering    the "CURRENT STATE" block the model reads
    objectives   objective catalogue, validation, post-reset verification
    served       served constants (crop economics, preset values) for the prompt
    tools        one wrapper per action plus the Ollama tool schemas
    prompt       the system message and tool-argument decoding
    agent        the LLM loop
    check_api    the scripted smoke test that needs no LLM
    reporting    latency plot, statistics and the _summary.json
    cli          main()
"""

__all__ = [
    "agent",
    "api",
    "budget",
    "check_api",
    "cli",
    "config",
    "episode",
    "objectives",
    "prompt",
    "rendering",
    "reporting",
    "runtime",
    "served",
    "tools",
    "tracking",
]
