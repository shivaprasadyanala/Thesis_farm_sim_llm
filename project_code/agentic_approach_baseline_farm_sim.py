"""
Agentic baseline for the farm simulation in ``game/phaser_test/farm_sim``.

This is the farm-sim counterpart of ``agentic_approach_baseline.py``: one
system prompt, one Ollama model, tool calls, and the same latency / token /
invalid-move / tool-sequence reporting. The difference is where the truth
lives:

    legacy baseline (port 3000)     farm-sim baseline (this file)
    ------------------------------  ------------------------------------------
    hand-maintained ``state`` dict  server state  (GET  /api/v1/state)
    python rules inside the tools   server rules  (POST /api/v1/action)
    posts its world back            never posts state, only actions
    hand-written tool list          tool catalogue (GET  /api/v1/tools)

The simulation keeps ticking on its own (500 ms per tick, 120 ticks per
in-game day), so the model can fast-forward with the ``advance`` tool.

The harness itself lives in the ``farm_agent`` package next to this file, one
module per concern:

    farm_agent/runtime.py      the module-level globals, in one place
    farm_agent/config.py       config.yaml, argparse, log file setup
    farm_agent/api.py          GET /api/v1/state, POST /api/v1/action
    farm_agent/tracking.py     tiles visited, jumps, revisits, per-episode reset
    farm_agent/budget.py       tick/minute math, the ``budget:`` line, advance clamp
    farm_agent/rendering.py    the "CURRENT STATE" block the model reads
    farm_agent/objectives.py   objective catalogue, validation, verification
    farm_agent/episode.py      reset / adopt an episode, resolve the task text
    farm_agent/served.py       the served constants the prompt may quote
    farm_agent/tools.py        the action wrappers + the tool schemas sent to the model
    farm_agent/prompt.py       the system message
    farm_agent/agent.py        the LLM loop, once per backend (Ollama, vLLM)
    farm_agent/check_api.py    the scripted smoke test (no LLM)
    farm_agent/reporting.py    latency plot, statistics, ``_summary.json``
    farm_agent/cli.py          main()

This file is only the entry point (``python -m farm_agent`` runs the same
``main``); every command below is unchanged.



Usage (run from this folder; the farm-sim server must be running):

    # terminal 1 - start the sim on 3111, the port this repo's config uses
    cd "../game/phaser_test/farm_sim"
    $env:PORT=3111; npm start                  # PowerShell -> http://127.0.0.1:3111
    PORT=3111 npm start                        # bash, same port

    # terminal 2
    python agentic_approach_baseline_farm_sim.py --model qwen3:8b --preset easy --seed 1

    # smoke-test the HTTP layer and the tools without an LLM:
    python agentic_approach_baseline_farm_sim.py --check-api

    # same loop against an OpenAI-compatible vLLM server instead of Ollama
    # (base_url/api_key/model/thinking come from the ``vllm:`` block of
    # config.yaml, and ``--model`` overrides ``vllm.model``):
    python agentic_approach_baseline_farm_sim.py --vllm --preset easy --seed 1

The base URL is taken from ``--url``, else from ``server_urls.farm_sim_url`` in
``config.yaml`` (``http://127.0.0.1:3111``), else the built-in default
``http://127.0.0.1:3111``. A bare ``npm start`` serves port 3100, hence the
explicit ``PORT=3111`` above; ``--url`` overrides both.

The task handed to the model defaults to the objective the preset assigns
(``sim/systems/objective.js``): ``--preset easy`` -> "Plant 2 crops and keep them
watered.", ``normal`` -> "Harvest 3 crops before the end of day 10.", ``hard`` ->
"Earn $150 from sales and harvest 4 crops before day 18.". The prompt, the first
user turn and ``world.task_text`` therefore all ask for exactly what the server
scores; ``--task "..."`` only rewrites that wording.

``--objective <id>`` decouples the two: the preset keeps choosing the *world*
(money, seeds, tank, weather, pests, rovers, fog) while the objective supplies the
*goal*, deadline and reward. ``--preset easy --objective harvest_normal`` is the
easy farm scored on "Harvest 3 crops before the end of day 10" (+$60), which is
how a run holds the task fixed and varies the difficulty (or the reverse).
``--list-objectives`` prints the ids the server accepts and the default each
preset assigns; the ids come from ``GET /api/v1/objectives``, so this script never
hard-codes them. Any preset accepts any id - the pairing is validated here because
an unknown id *silently* falls back to ``starter_easy`` server-side.

Outputs (``--log-folder``, default ``../../experiments/test_logs_baseline_farm_sim``):
    <model>_<preset>_seed<seed>.log          textual trace, same logger as the baseline
                                             (rewritten per run - no append)
    <model>_<preset>_seed<seed>_latency.png  wall-clock time per successful tool call,
                                             measured from the start of the LLM turn
                                             (same definition as the legacy baseline)
    <model>_<preset>_seed<seed>_summary.json machine-readable metrics of the episode,
                                             including ``stop_reason``
                                             (objective_complete / objective_failed /
                                             max_steps / silent_steps / llm_error) and
                                             the ``max_advance_ticks`` the run used

Guards that keep a run comparable to its log:
    * the episode's objective is verified against ``--objective`` (or the preset
      default) right after the reset - a mismatch aborts with exit code 3 instead
      of scoring a goal the summary does not name;
    * one ``advance`` call is capped at ``--max-advance-ticks`` (default 120 = one
      in-game day, 0 = off) and never runs past the objective deadline; a clipped
      call says so in ``note`` so the model does not assume the clock jumped;
    * an action that reports success without changing anything (``steps: 0``) is
      recorded as ``NO_PROGRESS`` and answered with the fix (rest), and three
      identical tool calls in a row earn a corrective user message;
    * every observation carries a ``budget:`` line (steps left, in-game days to the
      deadline, the per-call advance cap) and the ``advance`` tool description
      states that cap, because the recorded runs lost on both numbers being
      invisible: the model asked for 4-day jumps it never got and never knew how
      many steps or days it still had.
"""

import sys

from farm_agent.cli import main

if __name__ == "__main__":
    sys.exit(main())

