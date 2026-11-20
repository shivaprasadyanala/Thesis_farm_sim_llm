"""Mutable run state shared by every module of the harness.

The harness used to be a single script whose functions exchanged data through
module-level globals (``state``, ``invalid_moves``, ``MAX_STEPS``, ...). That
only survives a split into modules if each of those names exists exactly once
and is always reached as an attribute: a ``state = {}`` inside ``rendering.py``
would be a different object from the one ``tools.py`` writes. Every module
therefore does ``from . import runtime`` and touches ``runtime.<name>``, and
assignment works because ``runtime.state = ...`` and ``runtime.jumps += 1`` are
attribute writes on the one module object.

Nothing here imports another module of the package - this file is the bottom of
the dependency graph on purpose.
"""

import requests

#: `config.yaml` as a dict. Loaded once by `farm_agent.config.load_config`
#: (kept here because the Ollama URL in `server_urls` is read from several
#: modules).
config_data = {}

# ---------------------------------------------------------------------------
# HTTP layer - the only way this harness touches the game
# ---------------------------------------------------------------------------
#: Base URL of the farm-sim server, set by `farm_agent.cli.main`
#: (`--url`, else `server_urls.farm_sim_url` of config.yaml, else :3111).
base_url = ""
session = requests.Session()
session.headers.update({"Content-Type": "application/json"})

# One `advance` call may not skip more than a day of farm time and never runs past
# the objective deadline (`clamp_advance`). `--max-advance-ticks 0` removes the cap
# and leaves the raw server limit (MAX_TICKS_PER_ADVANCE = 5000 ticks = ~41 days).
MINUTES_PER_TICK = 12
MINUTES_PER_DAY = 24 * 60
MAX_ADVANCE_TICKS = 120
# Identical tool calls in a row before the harness tells the model it is stuck.
STUCK_REPEATS = 3
# Step budget (`--max-steps`) and the step the harness is on. The observation reports
# what is left, so the model does not have to count its own turns to know.
MAX_STEPS = 60
STEPS_USED = 0

# `GET /api/v1/config` read once per run: the served crop economics, preset
# constants and tick length. The prompt quotes these instead of literals that
# drift (it asked for "2 L per watering" for a day after the rebalance to 4 L,
# and never told the model how many kg a harvest is worth).
server_config = {}

# ---------------------------------------------------------------------------
# Episode bookkeeping (same counters as the legacy baseline)
# ---------------------------------------------------------------------------
state = {}                 # latest observation returned by the server
visited = set()            # tiles the farmer has stood on
player_positions = []      # one entry per tool call
previous_tile = None
invalid_moves = 0
invalid_move_object = {}
revisits = 0
jumps = 0
tool_calls = []            # names of the tool calls that succeeded
time_taken = []            # wall-clock seconds per successful tool call
total_input_tokens = 0
total_output_tokens = 0
# One chat completion per model *turn* - the unit `--max-steps` bounds - and every
# tool call those turns asked for. The two are reported separately because a turn
# can order more than one call while `no_of_llm_calls` counts turns, not calls.
llm_calls = 0
tool_calls_sent = 0
stop_reason = ""           # how the last episode ended (written to the summary)

# ---------------------------------------------------------------------------
# Objective selection - what the episode is scored on
# ---------------------------------------------------------------------------
#: Objective catalogue of the running server, filled by
#: `farm_agent.objectives.load_objective_catalogue`.
objective_catalog = []
preset_objectives = {}
