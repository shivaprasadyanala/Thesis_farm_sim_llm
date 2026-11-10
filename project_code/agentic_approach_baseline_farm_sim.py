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

Usage (run from this folder; the farm-sim server must be running):

    # terminal 1 - start the sim on 3111, the port this repo's config uses
    cd "../game/phaser_test/farm_sim"
    $env:PORT=3111; npm start                  # PowerShell -> http://127.0.0.1:3111
    PORT=3111 npm start                        # bash, same port

    # terminal 2
    python agentic_approach_baseline_farm_sim.py --model qwen3:8b --preset easy --seed 1

    # smoke-test the HTTP layer and the tools without an LLM:
    python agentic_approach_baseline_farm_sim.py --check-api

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

import argparse
import json
import logging
import math
import os
import re
import sys
import time

import matplotlib
matplotlib.use("Agg")  # experiment runs are headless
import matplotlib.pyplot as plt
import numpy as np
import requests
import yaml
from ollama import Client
from ollama._types import ChatResponse

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_config(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)  # safe_load prevents code execution
            return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error reading file: {e}")
        sys.exit(1)


config_data = read_config(os.path.join(BASE_DIR, "config.yaml"))

# Speech input is kept for parity with the legacy baseline, but the import is
# guarded: `speech_to_text` opens the microphone as soon as `config.yaml` has
# `speech.user_input: true`, and it needs `speech_recognition` installed.
try:
    from speech_to_text import audio_text  # noqa: F401
except Exception:  # pragma: no cover - optional dependency
    audio_text = ""

#: Default task text handed to the model. `None` means "use the objective this
#: preset assigns" (`game/phaser_test/farm_sim/sim/systems/objective.js`), so the
#: prompt always asks for exactly what the server scores:
#:   easy   -> starter_easy    "Plant 2 crops and keep them watered."
#:   normal -> harvest_normal  "Harvest 3 crops before the end of day 10."
#:   hard   -> profit_hard     "Earn $150 from sales and harvest 4 crops before day 18."
#: `--objective <id>` replaces the goal while keeping the preset's world, and
#: `--task` only rewrites the instruction text (the objective stays authoritative).
default_task = None


#: Tool sequence a competent baseline run produces at least once (see the
#: reporting section at the bottom of this file).
correct_seq = ['till', 'plant_crop', 'collect_water', 'water', 'harvest', 'sell']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Agentic baseline for the farm simulation (farm-sim REST API).",
    )
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model name served by Ollama (required unless --check-api)")
    parser.add_argument("--url", type=str, default=None,
                        help="farm-sim base URL (default: config server_urls.farm_sim_url "
                             "or http://127.0.0.1:3111)")
    parser.add_argument("--preset", type=str, default="easy",
                        choices=["easy", "normal", "hard"], help="difficulty preset of the episode")
    parser.add_argument("--objective", type=str, default=None,
                        help="objective id from the server (see --list-objectives), e.g. "
                             "starter_easy, harvest_normal, profit_hard; it replaces the goal "
                             "only - the preset still chooses the world "
                             "(default: the objective the preset assigns)")
    parser.add_argument("--list-objectives", action="store_true",
                        help="print the objective catalogue of the server and exit")
    parser.add_argument("--seed", type=int, default=1, help="world seed of the episode")
    parser.add_argument("--task", type=str, default=default_task,
                        help="task text given to the model and stored in the sim "
                             "(default: the objective the preset assigns)")
    # 60, not 50: the recorded `hard` episode's critical path was 54 steps (plant
    # day 1 -> two corn harvests -> carry -> two `sell`s -> the second wheat
    # harvest -> its sale) and it ended at step 50 holding $167 of unsold produce,
    # four steps short. `normal` finishes in 30-45, so the extra headroom is only
    # spent on the preset that needs it.
    parser.add_argument("--max-steps", type=int, default=60, help="maximum number of LLM steps")
    parser.add_argument("--llm-timeout", type=float, default=60.0,
                        help="seconds to wait for one LLM reply (default 60). Raise it when the "
                             "Ollama host has fallen back to CPU: a 26B model then needs longer "
                             "than a minute just to evaluate the prompt, and the run dies as "
                             "`llm_error` with `size_vram: 0` in /api/ps")
    parser.add_argument("--log-folder", type=str,
                        default="../experiments/test_logs_baseline_farm_sim",
                        help="folder for the .log/.png/.json artefacts")
    parser.add_argument("--tag", type=str, default="", help="extra suffix for the artefact names")
    parser.add_argument("--no-reset", action="store_true",
                        help="keep the episode that is already running on the server")
    parser.add_argument("--reset-after", action="store_true",
                        help="start another fresh episode after the run (leaves the game clean)")
    parser.add_argument("--max-advance-ticks", type=int, default=120,
                        help="cap one `advance` call at this many ticks (120 = one in-game "
                             "day, 0 = no cap); the harness also never advances past the "
                             "objective deadline")
    parser.add_argument("--check-api", action="store_true",
                        help="skip the LLM and drive the API with a fixed script (smoke test)")
    args = parser.parse_args(argv)
    if not args.check_api and not args.list_objectives and not args.model:
        parser.error("--model is required unless --check-api or --list-objectives is used")
    if args.max_advance_ticks < 0:
        parser.error("--max-advance-ticks must be >= 0 (0 turns the cap off)")
    return args


def resolve_base_url(args):
    if args.url:
        return args.url.rstrip("/")
    configured = (config_data.get("server_urls") or {}).get("farm_sim_url")
    if configured:
        return str(configured).rstrip("/")
    return "http://127.0.0.1:3111"


def setup_logging(args):
    """Same log layout as the legacy baseline: one file per model/preset/seed.

    `filemode='w'` on purpose: the default `'a'` appended every run into the same
    `model_preset_seed.log`, so the recorded logs mixed up to 20 episodes (and
    their objectives) in one file - impossible to attribute after the fact. The
    artefacts of a run are now one `.log`, one `_summary.json` and one `.png`.
    """
    os.makedirs(args.log_folder, exist_ok=True)
    stem = f"{args.model or 'check_api'}_{args.preset}_seed{args.seed}{args.tag}"
    stem = stem.replace(":", "_").replace(".", "_").replace("/", "_").replace("\\", "_")
    log_path = os.path.join(args.log_folder, f"{stem}.log")
    logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.INFO,
                        filemode='w',
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("httpx").disabled = True
    logging.getLogger("httpcore").disabled = True
    return stem, log_path


# ---------------------------------------------------------------------------
# HTTP layer - the only way this script touches the game
# ---------------------------------------------------------------------------
BASE_URL = ""
SESSION = requests.Session()
SESSION.headers.update({"Content-Type": "application/json"})

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
SERVER_CONFIG = {}


def api_get(path, **params):
    response = SESSION.get(f"{BASE_URL}{path}", params=params or None, timeout=10)
    response.raise_for_status()
    return response.json()


def api_post(path, payload, include_state=False):
    """POST one action. `include_state=1` echoes the fresh observation back."""
    params = {"include_state": 1} if include_state else None
    response = SESSION.post(f"{BASE_URL}{path}", json=payload, params=params, timeout=30)
    response.raise_for_status()
    return response.json()


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
stop_reason = ""           # how the last episode ended (written to the summary)


def reset_tracking():
    global state, visited, player_positions, previous_tile, invalid_moves
    global invalid_move_object, revisits, jumps, tool_calls, time_taken
    global total_input_tokens, total_output_tokens, stop_reason, STEPS_USED
    state = {}
    visited = set()
    player_positions = []
    previous_tile = None
    invalid_moves = 0
    invalid_move_object = {}
    revisits = 0
    jumps = 0
    tool_calls = []
    time_taken = []
    total_input_tokens = 0
    total_output_tokens = 0
    stop_reason = ""
    STEPS_USED = 0


def refresh_state(events=6):
    global state
    state = api_get("/api/v1/state", events=events)
    return state


def prepare_episode(args):
    """Start a fresh episode (or adopt the running one) and track its start tile."""
    global state, previous_tile
    reset_tracking()
    if args.no_reset:
        refresh_state()
        print(f"keeping running episode: preset={state['preset']} seed={state['seed']} "
              f"tick={state['tick']} objective={state['objective']['id']}")
        logger.info(f"episode kept: preset={state['preset']} seed={state['seed']} "
                    f"tick={state['tick']} objective={state['objective']['id']}")
        # A paused world rejects every action with `PAUSED`, which looks exactly
        # like an agent that does nothing - resume it before the run.
        if state["time"].get("paused"):
            api_post("/api/v1/pause", {"paused": False})
            refresh_state()
            print("the running episode was paused - resumed it so the agent can act")
            logger.info("episode was paused, resumed for the run")
    else:
        payload = {"preset": args.preset, "seed": args.seed}
        if args.objective:
            # The goal is selected independently of the world: `--preset easy
            # --objective harvest_normal` is the easy farm scored on the harvest.
            payload["objective"] = args.objective
        if args.task:
            # No `task` key means the server keeps the objective text as
            # `world.task_text`, so the prompt and the HUD ask for the same thing.
            payload["task"] = args.task
        result = api_post("/api/v1/reset", payload)
        state = result["state"]
        print(f"episode {result['episode']} preset={result['preset']} seed={result['seed']} "
              f"objective={state['objective']['id']}")
        logger.info(f"episode reset: episode={result['episode']} preset={result['preset']} "
                    f"seed={result['seed']} objective={state['objective']['id']}")
    previous_tile = (int(state["player"]["tile"][0]), int(state["player"]["tile"][1]))
    visited.add(previous_tile)
    player_positions.append(list(previous_tile))
    # Never let a stale goal survive into the run: the summary and the log have to
    # describe the objective the server is actually scoring (`--no-reset` and an
    # unknown id are the two ways a previous run's goal used to leak in).
    exit_code = verify_episode_objective(args)
    if exit_code:
        raise SystemExit(exit_code)
    return state


def resolve_task(args):
    """Task text the prompt asks for: `--task`, else the episode's objective.

    `state["task"]` already mirrors the objective unless `--task` overrode it, so
    the system prompt, the first user message and the HUD all agree with the
    objective the server scores. Without this the prompt's task block was empty
    and the model only saw the objective buried in the state dump.
    """
    if args.task:
        return args.task
    objective = state.get("objective") or {}
    text = objective.get("text") or state.get("task") or "farm the field"
    deadline = objective.get("deadline_day", "?")
    return f"{text} The episode ends when the objective is complete (it fails after day {deadline})."


def track_position(action):
    """Record the farmer tile after an action: revisits and unexplained jumps."""
    global previous_tile, jumps, revisits
    if not state:
        return
    tile = (int(state["player"]["tile"][0]), int(state["player"]["tile"][1]))
    if previous_tile is not None and tile != previous_tile:
        step = abs(tile[0] - previous_tile[0]) + abs(tile[1] - previous_tile[1])
        if step > 1 and action != "move_to":
            # `move_to` walks a whole A* path, so only other actions count.
            jumps += 1
            print(f"Jump of {step} tiles: {previous_tile} -> {tile} ({action})")
        if tile in visited:
            revisits += 1
    visited.add(tile)
    previous_tile = tile
    player_positions.append([tile[0], tile[1]])


# ---------------------------------------------------------------------------
# State rendering - the "CURRENT STATE" block the model reads
# ---------------------------------------------------------------------------
CROP_LETTERS = {"wheat": "W", "rice": "R", "corn": "C", "tomato": "T"}


def describe_plot(plot):
    tile = f"({plot['x']},{plot['y']})"
    if not plot.get("known", True):
        return f"{plot['id']} {tile} out of sight (fog of war) - use inspect or scan"
    crop = plot.get("crop")
    if not crop:
        return f"{plot['id']} {tile} tilled, empty, moisture {plot['moisture']:.2f}"
    if crop.get("dead"):
        progress = f"DEAD ({crop.get('dead_reason') or 'unknown'})"
    elif crop.get("ready"):
        progress = "READY to harvest"
    else:
        progress = (f"{crop.get('stage')} growth {crop.get('growth', 0):.2f}, "
                    f"ripe in {crop.get('days_to_ripe')}d")
    flags = [
        progress,
        f"moisture {plot['moisture']:.2f}",
        "needs water" if plot.get("needs_water") else "watered",
    ]
    if crop.get("weeds"):
        flags.append(f"weeds {crop['weeds']}")
    if crop.get("disease"):
        flags.append(f"disease {crop['disease']}")
    return f"{plot['id']} {tile} {crop['type']}: " + ", ".join(flags)


def plots_to_text(current, max_details=8):
    """A small ASCII map of the field plus one line per worked plot."""
    plots = [plot for plot in current.get("plots", []) if plot.get("known")]
    if not plots:
        return "  (no plot data - fog of war)"

    columns = sorted({plot["x"] for plot in plots})
    rows = sorted({plot["y"] for plot in plots})
    at = {(plot["x"], plot["y"]): plot for plot in plots}

    lines = [
        f"  field grid (y = row, x = {columns[0]}..{columns[-1]}): "
        ". empty  t tilled  W wheat  R rice  C corn  T tomato  X dead"
    ]
    for y in rows:
        cells = []
        for x in columns:
            plot = at.get((x, y))
            if plot is None:
                cells.append("?")
                continue
            crop = plot.get("crop")
            if crop:
                cells.append("X" if crop.get("dead") else CROP_LETTERS.get(crop["type"], "c"))
            else:
                cells.append("t" if plot.get("tilled") else ".")
        lines.append(f"  y={y:>2}  " + " ".join(cells))

    worked = sorted(
        (plot for plot in plots if plot.get("crop") or plot.get("tilled")),
        key=lambda plot: (plot.get("distance", 0), plot["id"]),
    )
    lines.append(f"  worked plots ({len(worked)}/{len(plots)}):")
    if not worked:
        lines.append("    none - till a plot first")
    for plot in worked[:max_details]:
        lines.append("    " + describe_plot(plot))
    if len(worked) > max_details:
        lines.append(f"    ... {len(worked) - max_details} more (use scan or inspect)")

    needs_water = sum(1 for plot in plots if plot.get("needs_water"))
    ripe = sum(1 for plot in plots if plot.get("crop") and plot["crop"].get("ready"))
    lines.append(f"  dry plots: {needs_water} | ripe plots: {ripe}")
    return "\n".join(lines)


def state_to_text(current):
    """Compact digest of the observation handed to the model after every action.

    Ends with the field map and a ``budget:`` line (steps left, game days to the
    deadline, per-call advance cap) - see `budget_to_text`.
    """
    if not current:
        return "  (state not loaded yet)"
    player = current["player"]
    clock = current["time"]
    weather = current["weather"]
    silo = current["silo"]
    objective = current["objective"]
    market = current.get("market", {})

    silo_items = ", ".join(f"{name} {item['kg']} kg" for name, item in silo["items"].items()) or "empty"
    status = "complete" if objective["is_complete"] else "failed" if objective["is_failed"] else "open"

    lines = [
        f"day {clock['day']} {clock['season']} {clock['clock']} | weather {weather['label']} "
        f"{weather['temp_c']}C, rain {weather['rain_rate']} | tick {current['tick']} "
        f"(speed {clock['speed']}, paused {clock['paused']})",
        f"player tile ({player['tile'][0]},{player['tile'][1]}) facing {player['facing']} | "
        f"energy {player['energy']}/{player['max_energy']} | money ${player['money']}",
        f"can {player['can']['litres']}/{player['can']['capacity']} L | "
        f"tank {current['tank']['litres']}/{current['tank']['capacity']} L | "
        f"silo {silo['total_kg']}/{silo['capacity_kg']} kg ({silo_items})",
        "seeds: " + ", ".join(f"{name} {count}" for name, count in player["seeds"].items()),
        "landmarks: " + ", ".join(f"{name} ({pos[0]},{pos[1]})"
                                  for name, pos in current["landmarks"].items()),
        "rocks: " + ", ".join(f"({rock['tile'][0]},{rock['tile'][1]})"
                              for rock in current.get("obstacles", [])),
        "prices: " + ", ".join(f"{name} ${price}" for name, price in market.get("prices", {}).items()),
        f"objective [{status}] {objective['label']}: {objective['text']} | "
        f"progress {json.dumps(objective['progress'])} | deadline day {objective['deadline_day']} | "
        f"reward ${objective['reward']}",
    ]

    # Budget before the map: the recorded `hard` run spent steps 45-49 of its 50 on
    # energy recovery and the `normal` run burned nine `advance` calls without
    # knowing it had two days to the deadline, so the two numbers the model has to
    # trade against each other are stated instead of left to be derived.
    budget = budget_to_text()
    if budget:
        lines.append(budget)

    robots = current.get("robots") or []
    if robots:
        lines.append("rovers: " + ", ".join(
            f"{robot['id']} at ({robot['tile'][0]},{robot['tile'][1]}) task {robot['task']} "
            f"battery {robot['battery']}" for robot in robots))

    lines.extend(["field:", plots_to_text(current)])

    events = current.get("events") or []
    if events:
        lines.append("last events: " + " | ".join(event["message"] for event in events[-3:]))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools - every tool is one POST /api/v1/action, nothing is simulated locally
# ---------------------------------------------------------------------------
def progressless(data):
    """True when an action claimed success but the farmer did not move at all.

    `move_to` answers `steps: 0` / `reached: false` when its walk loop stops on
    `energy < 1`. Counting that as a valid action hides the failure from the
    invalid-move table and leaves the model with no reason to `rest`, so the
    harness turns it into a failure with an actionable message.
    """
    return data.get("reached") is False and data.get("steps") == 0


def call_action(payload, note=None):
    """Send one action, refresh `state` and return the result as a JSON string.

    The reply keeps the shape the legacy tools used (`status` / `action` /
    `message`) so the same evaluation code can read both experiments, plus the
    server's machine-readable `reason` and `data`. A `note` (used by `advance` for
    a clipped jump) is merged into the reply *before* it is printed and logged, so
    the trace shows the same thing the model saw.
    """
    global state, invalid_moves
    action = str(payload.get("type", "?"))
    # Drop unset optional arguments: the server treats `null` as a value
    # (`move_to` with steps=null would only walk one tile, `water` with
    # tile=null targets the faced tile), so absent is not the same as null.
    payload = {key: value for key, value in payload.items() if value is not None}
    started = time.time()
    try:
        result = api_post("/api/v1/action", payload, include_state=True)
    except Exception as exc:
        invalid_moves += 1
        invalid_move_object["transport error"] = invalid_move_object.get("transport error", 0) + 1
        logger.error(f"tool: {action} failed over HTTP: {exc}")
        return json.dumps({"status": "false", "action": action, "message": f"transport error: {exc}"})

    if isinstance(result.get("state"), dict):
        state = result.pop("state")

    ok = bool(result.get("ok"))
    reply = {
        "status": "true" if ok else "false",
        "action": action,
        "message": result.get("message") or "",
        "data": result.get("data") or {},
    }
    if note:
        reply["note"] = note
    if ok and progressless(reply["data"]):
        # The reply said "walked 0 of N tiles" and reported success: nothing moved,
        # so the call must not be counted as a valid action nor handed back as one.
        ok = False
        reply["status"] = "false"
        reply["reason"] = "NO_PROGRESS"
        reply["message"] = (f"{reply['message']} Nothing moved - the farmer is out of "
                            "energy. Call `rest` before walking again.")
    if not ok:
        reason = reply.get("reason") or result.get("reason") or "UNKNOWN_ACTION"
        reply["reason"] = reason
        invalid_moves += 1
        invalid_move_object[reason] = invalid_move_object.get(reason, 0) + 1

    track_position(action)
    tool_time = time.time() - started
    print(f"tool {action} -> ok={ok} reason={reply.get('reason')} "
          f"message={reply['message']!r} ({tool_time:.3f}s)")
    logger.info(f"tool: {action} args={json.dumps({k: v for k, v in payload.items() if k != 'type'})} "
                f"result: {json.dumps(reply)} tool_time={tool_time:.3f}s")
    return json.dumps(reply)


def move(dir: str = None) -> str:
    """Move the farmer exactly one tile north/south/east/west.

    Args:
        dir: 'up', 'down', 'left' or 'right'.
    """
    return call_action({"type": "move", "dir": dir})


def move_to(tile=None, steps: int = None) -> str:
    """Walk to a tile with A* pathfinding (1 energy per tile).

    Args:
        tile: destination [x, y].
        steps: optional maximum number of tiles to walk.
    """
    return call_action({"type": "move_to", "tile": tile, "steps": steps})


def till(tile=None) -> str:
    """Till an empty plot so it can be planted (4 energy).

    Args:
        tile: plot tile [x, y]; defaults to the plot you stand on or face.
    """
    return call_action({"type": "till", "tile": tile})


def plant_crop(crop: str = None, tile=None) -> str:
    """Plant a seed on a tilled, empty plot within reach.

    Args:
        crop: 'wheat', 'rice', 'corn' or 'tomato'.
        tile: plot tile [x, y]; defaults to the plot you stand on or face.
    """
    return call_action({"type": "plant_crop", "crop": crop, "tile": tile})


def water(tile=None) -> str:
    """Water a plot from the watering can (one can charge of litres, 2 energy).

    Args:
        tile: plot tile [x, y]; defaults to the plot you stand on or face.
    """
    return call_action({"type": "water", "tile": tile})


def collect_water(liters: float = None) -> str:
    """Fill the watering can from the tank (stand next to the tank).

    Args:
        liters: litres to take (default: fill the can).
    """
    return call_action({"type": "collect_water", "liters": liters})


def harvest(tile=None) -> str:
    """Harvest a ripe crop into the silo.

    Args:
        tile: plot tile [x, y]; defaults to the plot you stand on or face.
    """
    return call_action({"type": "harvest", "tile": tile})


def clear(tile=None) -> str:
    """Remove a dead or unwanted crop so the plot can be reused.

    Args:
        tile: plot tile [x, y]; defaults to the plot you stand on or face.
    """
    return call_action({"type": "clear", "tile": tile})


def weed(tile=None) -> str:
    """Remove weeds from a planted plot (2 energy).

    Args:
        tile: plot tile [x, y]; defaults to the plot you stand on or face.
    """
    return call_action({"type": "weed", "tile": tile})


def treat(tile=None) -> str:
    """Treat a diseased plot for $4 and 2 energy.

    Args:
        tile: plot tile [x, y]; defaults to the plot you stand on or face.
    """
    return call_action({"type": "treat", "tile": tile})


def fertilize(tile=None) -> str:
    """Raise the soil fertility of a plot for $6 and 3 energy.

    Args:
        tile: plot tile [x, y]; defaults to the plot you stand on or face.
    """
    return call_action({"type": "fertilize", "tile": tile})


def scan(radius: int = None) -> str:
    """Read soil moisture, crop stage and water needs of the nearby plots.

    Args:
        radius: search radius in tiles (0-8, default 2).
    """
    return call_action({"type": "scan", "radius": radius})


def inspect(tile=None) -> str:
    """Detailed status of one tile or plot.

    Args:
        tile: tile [x, y]; defaults to the plot you stand on or face.
    """
    return call_action({"type": "inspect", "tile": tile})


def buy(crop: str = None, qty: int = None) -> str:
    """Buy seeds at the silo (stand within 2 tiles of it).

    Args:
        crop: 'wheat', 'rice', 'corn' or 'tomato'.
        qty: number of seeds.
    """
    return call_action({"type": "buy", "crop": crop, "qty": qty})


def sell(crop: str = None, kg: float = None, contract: str = None) -> str:
    """Sell stored crops at the silo (stand within 2 tiles of it).

    Args:
        crop: 'wheat', 'rice', 'corn' or 'tomato'.
        kg: kilograms to sell (default: everything of that crop).
        contract: optional contract id for a better price.
    """
    return call_action({"type": "sell", "crop": crop, "kg": kg, "contract": contract})


def dispatch(task: str = None, robot: str = None) -> str:
    """Put a rover on a standing task; it then acts on its own, tick after tick.

    Task 'water' is irrigation duty: the rover refills at the tank, waters the dry
    crop plots without another call, and recharges when its battery drops.

    Args:
        task: 'water', 'charge' or 'recall'.
        robot: rover id (default: the first rover; call once per rover).
    """
    return call_action({"type": "dispatch", "task": task, "robot": robot})


def rest() -> str:
    """Rest to restore energy (25 anywhere, full at the farmhouse)."""
    return call_action({"type": "rest"})


def requested_ticks(minutes, ticks):
    """Ticks a model asked for; `minutes` is converted with 12 minutes per tick."""
    if ticks is not None:
        try:
            return max(0, int(ticks))
        except (TypeError, ValueError):
            logger.info(f"advance: unusable ticks={ticks!r} - treated as 0 ticks")
            return 0
    if minutes is not None:
        try:
            return max(0, int(minutes) // MINUTES_PER_TICK)
        except (TypeError, ValueError):
            logger.info(f"advance: unusable minutes={minutes!r} - treated as 0 ticks")
            return 0
    return 0


def ticks_until_deadline():
    """Ticks left before the objective deadline day ends (None when unknown)."""
    objective = state.get("objective") or {}
    time_block = state.get("time") or {}
    deadline = objective.get("deadline_day")
    day, minute = time_block.get("day"), time_block.get("minute")
    if deadline is None or day is None or minute is None:
        return None
    remaining_minutes = (int(deadline) - int(day) + 1) * MINUTES_PER_DAY - int(minute)
    return max(0, remaining_minutes // MINUTES_PER_TICK)


def budget_to_text():
    """One line of budget telemetry: steps left, game time left, advance cap.

    Both numbers are known to the harness and were invisible to the model. In the
    recorded `hard` run it spent five of its last ten steps recovering energy
    without knowing only five were left; in `normal` it chained nine `advance`
    calls (each clipped to one day) and hit the deadline with twelve steps unused.
    """
    parts = []
    if MAX_STEPS > 0:
        parts.append(f"{max(0, MAX_STEPS - STEPS_USED)} of {MAX_STEPS} step(s) left")
    remaining = ticks_until_deadline()
    if remaining is not None:
        days = remaining * MINUTES_PER_TICK / MINUTES_PER_DAY
        deadline = (state.get("objective") or {}).get("deadline_day")
        parts.append(f"{days:.1f} day(s) to the deadline (day {deadline})")
    if MAX_ADVANCE_TICKS > 0:
        per_call = MAX_ADVANCE_TICKS * MINUTES_PER_TICK / MINUTES_PER_DAY
        parts.append(f"one advance = at most {MAX_ADVANCE_TICKS} tick(s) = {per_call:.1f} day(s) "
                     "and costs one step")
    return ("budget: " + " | ".join(parts)) if parts else ""


def clamp_advance(requested):
    """Keep one `advance` inside `--max-advance-ticks` and inside the deadline.

    Returns the ticks to actually send plus a note ('' when nothing was clipped)
    that is echoed to the model as `note`, so it can see the clock did not jump
    as far as it asked instead of assuming it did.
    """
    limit, source = None, ""
    if MAX_ADVANCE_TICKS > 0:
        limit, source = MAX_ADVANCE_TICKS, f"--max-advance-ticks {MAX_ADVANCE_TICKS}"
    to_deadline = ticks_until_deadline()
    if to_deadline is not None and (limit is None or to_deadline < limit):
        limit = to_deadline
        source = f"objective deadline (day {state['objective']['deadline_day']})"
    if limit is None or requested <= limit:
        return requested, ""
    allowed = max(0, limit)
    note = f"capped by {source}"
    if allowed > 0:
        # Say what the clipping costs in calls: the model kept asking for 480/360
        # ticks and assumed the clock had jumped three days, so it planned one
        # step where three were needed.
        extra = (requested - allowed + allowed - 1) // allowed
        note += (f" - {extra} more call(s) of {allowed} tick(s) reach the requested "
                 f"{requested}; each call costs one step")
    return allowed, note


def advance(minutes: int = None, ticks: int = None) -> str:
    """Fast-forward the simulation (1 tick = 12 simulated minutes, 120 ticks = 1 day).

    The call is capped at `--max-advance-ticks` (default 120 = one farm day) and it
    never runs past the objective deadline. A model that reads "one day" as
    `ticks=1440` would otherwise skip 12 days and fail a day-10 objective in a
    single call, which is how the recorded `normal` run lost its episode.

    Args:
        minutes: simulated minutes to skip.
        ticks: ticks to skip.
    """
    requested = requested_ticks(minutes, ticks)
    allowed, note = clamp_advance(requested)
    if note:
        print(f"advance capped: requested {requested} tick(s), applied {allowed} ({note})")
        logger.info(f"advance capped: requested {requested} tick(s), applied {allowed} ({note})")
    # `note` goes in through `call_action` so the clipped reply is printed and
    # logged with the note, not before it (the .log used to show a plain result).
    return call_action({"type": "advance", "ticks": allowed}, note=note)


available_tools = {
    "move": move,
    "move_to": move_to,
    "till": till,
    "plant_crop": plant_crop,
    "water": water,
    "collect_water": collect_water,
    "harvest": harvest,
    "clear": clear,
    "weed": weed,
    "treat": treat,
    "fertilize": fertilize,
    "scan": scan,
    "inspect": inspect,
    "buy": buy,
    "sell": sell,
    "dispatch": dispatch,
    "rest": rest,
    "advance": advance,
}


def fetch_tool_schemas():
    """Read the tool catalogue from the server and convert it to Ollama tools.

    `sim/tools.js` is the single source of truth for the action surface, so the
    model is offered exactly what the running build supports.
    """
    catalogue = api_get("/api/v1/tools")["tools"]
    schemas = []
    for tool in catalogue:
        schemas.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": {
                    "type": "object",
                    "properties": tool.get("parameters") or {},
                    "required": tool.get("required") or [],
                },
            },
        })
    # The server advertises `advance` with no cap, and the model trusts the schema
    # over the prompt: it kept asking for 480 ticks (4 days) and read the answer as
    # "4 days passed". State the cap where the model actually reads it.
    if MAX_ADVANCE_TICKS > 0:
        per_call = MAX_ADVANCE_TICKS * MINUTES_PER_TICK / MINUTES_PER_DAY
        for entry in schemas:
            if entry["function"]["name"] == "advance":
                entry["function"]["description"] = (
                    f"{entry['function']['description']} Capped: one call moves at most "
                    f"{MAX_ADVANCE_TICKS} ticks ({per_call:.0f} farm day), never past the "
                    "objective deadline, and costs one step - a 3-day wait is 3 calls, so "
                    "water dry plots before waiting.")
    missing = [entry["function"]["name"] for entry in schemas
               if entry["function"]["name"] not in available_tools]
    if missing:
        raise RuntimeError(f"server advertises tools without a local wrapper: {missing}")
    return schemas


# ---------------------------------------------------------------------------
# Objective selection - what the episode is scored on
# ---------------------------------------------------------------------------
#: Objective catalogue of the running server, filled by `load_objective_catalogue`.
objective_catalog = []
preset_objectives = {}


def load_objective_catalogue():
    """Objectives the server accepts, and the one each preset assigns by default.

    ``POST /api/v1/reset {objective}`` takes any id from
    ``sim/systems/objective.js``, so the catalogue is read from the server instead
    of being duplicated here. A server started before ``GET /api/v1/objectives``
    existed still answers ``/api/v1/config``, whose presets name their default
    objective - which is enough to validate a choice.
    """
    global objective_catalog, preset_objectives
    objective_catalog = []
    preset_objectives = {}
    try:
        objective_catalog = api_get("/api/v1/objectives")["objectives"]
    except Exception as exc:
        print(f"note: this server does not list its objectives ({exc}); "
              f"falling back to the preset defaults of /api/v1/config")
    try:
        presets = api_get("/api/v1/config")["presets"]
        preset_objectives = {name: preset.get("objective") for name, preset in presets.items()}
    except Exception as exc:
        print(f"note: cannot read the preset catalogue either ({exc})")
    if not objective_catalog:
        ids = sorted({obj_id for obj_id in preset_objectives.values() if obj_id})
        objective_catalog = [{"id": obj_id, "text": "", "deadline_day": "?", "reward": "?"}
                             for obj_id in ids]
    return objective_catalog


def print_objectives():
    """`--list-objectives`: what the server can score and which preset picks what."""
    load_objective_catalogue()
    if not objective_catalog:
        print("the server lists no objectives - is the farm-sim server reachable?")
        return
    print(f"{'id':<16} {'deadline':>8}  {'reward':>6}  objective")
    for entry in objective_catalog:
        print(f"{entry['id']:<16} {str(entry.get('deadline_day', '?')):>8}  "
              f"${str(entry.get('reward', '?')):>5}  {entry.get('text', '')}")
    print("")
    print("default objective per preset:")
    for name, obj_id in preset_objectives.items():
        print(f"  {name:<8} {obj_id}")
    print("")
    print("any preset accepts any id, e.g. --preset easy --objective harvest_normal")
    print("(the preset then only decides the world: money, seeds, weather, pests, rovers)")


def validate_objective(args):
    """Reject an unknown `--objective` before anything is reset.

    The server falls back to ``starter_easy`` for an unknown id (`objective.js`),
    which would make a run silently score a different goal than the one asked for.
    """
    if not args.objective:
        return 0
    if not objective_catalog:
        load_objective_catalogue()
    ids = [entry["id"] for entry in objective_catalog]
    if not ids:
        print(f"note: cannot validate --objective {args.objective}; the server decides")
        return 0
    if args.objective not in ids:
        print(f"unknown objective: {args.objective}")
        print(f"available: {', '.join(ids)}")
        print("list them with: --list-objectives")
        return 2
    return 0


def describe_objective(args):
    """One line that makes the preset/objective mix of a run unambiguous."""
    default = preset_objectives.get(args.preset) or "?"
    if not args.objective:
        return f"objective: {default} (the {args.preset} preset default)"
    line = f"objective: {args.objective} (requested; {args.preset} preset default: {default})"
    if args.task:
        line += (f"\nnote: --task only rewrites the instruction text - "
                 f"{args.objective} is what the server scores")
    return line


def verify_episode_objective(args):
    """Check that the running episode scores the goal this run claims to score.

    Called from `prepare_episode`, i.e. after `/api/v1/reset` or after adopting a
    running episode with `--no-reset`. It is the guard against the defect that
    made the recorded logs unusable: a run whose summary named one objective while
    the server was scoring another (a leftover episode, or an id the server did
    not know and replaced with `starter_easy`).
    """
    actual = (state.get("objective") or {}).get("id")
    if args.objective:
        expected = args.objective
    elif args.no_reset:
        expected = None          # the running episode decides; --objective is the only contract
    else:
        expected = preset_objectives.get(args.preset)
    if not actual:
        print("warning: the state carries no objective id - cannot verify the goal")
        logger.warning("no objective id in the state - cannot verify the goal")
        return 0
    if expected and actual != expected:
        print(f"objective mismatch: the episode scores {actual}, this run expects {expected}")
        print("start a fresh episode (drop --no-reset) or pass a matching --objective")
        logger.error(f"objective mismatch: episode={actual} expected={expected} "
                     f"requested={args.objective or 'preset default'}")
        return 3
    source = "requested" if args.objective else ("running episode" if args.no_reset else "preset default")
    print(f"objective verified: {actual} ({source})")
    logger.info(f"objective verified: {actual} ({source}); requested="
                f"{args.objective or 'preset default'}")
    return 0


# ---------------------------------------------------------------------------
# Served constants - what the prompt is allowed to claim
# ---------------------------------------------------------------------------
def load_server_config():
    """Cache ``GET /api/v1/config`` for the prompt; never fatal.

    Everything the model is told about payoffs, litres and tick length comes from
    here, so a rebalanced sim reaches the prompt without a code edit. This is what
    the recorded `normal` run lacked: it planted four wheats - a crop whose four
    harvests gross about $115 - against a $150 objective, and its trace says
    "The harvest amount is not specified in the prompt" even though the server
    publishes `yield_kg` and `sell_price` for every crop.
    """
    global SERVER_CONFIG, MINUTES_PER_TICK, MINUTES_PER_DAY
    try:
        SERVER_CONFIG = api_get("/api/v1/config") or {}
    except requests.RequestException as exc:
        logger.warning(f"/api/v1/config unavailable ({exc}); prompt falls back to literal figures")
        SERVER_CONFIG = {}
        return
    tick_minutes = SERVER_CONFIG.get("minutes_per_tick")
    ticks_per_day = SERVER_CONFIG.get("ticks_per_day")
    if isinstance(tick_minutes, (int, float)) and tick_minutes > 0:
        MINUTES_PER_TICK = tick_minutes
    if isinstance(ticks_per_day, (int, float)) and ticks_per_day > 0 and tick_minutes:
        # Derived, not assumed: "one advance = one day" is only true if the server
        # still runs 120 ticks of 12 minutes.
        MINUTES_PER_DAY = tick_minutes * ticks_per_day


def preset_value(name):
    """One served preset field for the episode's preset (None when absent)."""
    preset_name = (state or {}).get("preset")
    if not preset_name:
        preset_name = (SERVER_CONFIG.get("current") or {}).get("preset")
    preset = (SERVER_CONFIG.get("presets") or {}).get(preset_name) or {}
    return preset.get(name)


def preset_constant(name):
    """One served preset constant for the episode's preset (None when absent)."""
    value = preset_value(name)
    return value if isinstance(value, (int, float)) else None


def litres_per_watering():
    """Litres one `water` action pours, or None when the server does not say."""
    return preset_constant("water_per_action")


def days_per_season():
    """Season length in days, or None when the server does not publish it.

    The `hard` preset pins `growth_speed` at 1.0 and turns to winter on day 13 of
    an 18-day deadline, so a model that does not know a season is four days long
    plants corn into a cold snap and harvests nothing.
    """
    value = SERVER_CONFIG.get("days_per_season")
    return value if isinstance(value, (int, float)) else None


def starting_wallet_text():
    """`$28 (you also hold wheat 2, rice 2, already paid for)` - preset-sized.

    The recorded `hard` run tried to buy four corn seeds ($28) with $20, got
    NO_MONEY, then bought two and could no longer afford a full field. The wallet
    and the free seeds are therefore stated together with what seed costs.
    """
    money = preset_constant("starting_money")
    seeds = preset_value("starting_seeds")
    seeds = seeds if isinstance(seeds, dict) else {}
    seeds_text = ", ".join(f"{name} {count}" for name, count in seeds.items() if count)
    wallet = f"${money:g}" if isinstance(money, (int, float)) else "limited"
    if money is None and not seeds_text:
        return ""
    if seeds_text:
        return f"{wallet} + {seeds_text} already held"
    return wallet



def crop_economics_to_text():
    """One line per crop: days, seed cost, price per kg, yield, gross $ and seasons.

    Without the yield the objective is unplannable: `$150 from 4 harvests` is
    impossible for wheat (~$29 each) and trivial for corn (~$75 each), and only
    the server knows which is which. The seasons column is served by
    `catalogView` and is what makes the `hard` preset's winter stall visible
    before the crop stops growing instead of after.
    """
    crops = SERVER_CONFIG.get("crops") or {}
    if not crops:
        # Pre-config fallback: the shape of the catalogue, not the numbers.
        return ("crops: wheat, rice, corn, tomato - check `inspect` after planting;"
                " `sell` prices are per kg")
    rows = ["crops (days to ripen | seed $ | sell $/kg | yield kg per harvest"
            " | gross $ per harvest | seasons it grows in):"]
    for name, crop in crops.items():
        days = crop.get("days", "?")
        seed_cost = crop.get("seed_cost", "?")
        price = crop.get("sell_price", "?")
        yield_kg = crop.get("yield_kg", 0) or 0
        gross = yield_kg * (price if isinstance(price, (int, float)) else 0)
        rows.append(f"  {name:<6} {days:>4}d | ${seed_cost:<2} | ${price:<4}/kg | "
                    f"{yield_kg:>4} kg | ~${gross:>4.0f} | {crop_seasons_text(crop)}")
    return "\n".join(rows)


#: Season codes for the crop table: four letters beat "spring,summer,autumn" in a
#: table the model re-reads on every step.
_SEASON_CODES = {"spring": "spr", "summer": "sum", "autumn": "aut", "winter": "win"}


def crop_seasons_text(crop):
    """`spr,sum,aut` - the seasons a crop grows in (served by `catalogView`)."""
    seasons = crop.get("seasons") or []
    codes = [_SEASON_CODES.get(season, season[:3]) for season in seasons]
    return ",".join(codes) if codes else "?"


def objective_targets():
    """(earnings, harvests) the served objective text asks for; (None, None) if unknown.

    `/api/v1/config` publishes the objective's prose (`Earn $150 from sales and
    harvest 4 crops before day 18`) but not its numbers, so the two figures the
    plan needs are read out of that prose. Anything unparsable yields None and the
    PLAN block is shortened instead of guessing a target.
    """
    text = ((state or {}).get("objective") or {}).get("text") or ""
    earnings = harvests = None
    money_match = re.search(r"\$(\d+(?:\.\d+)?)", text)
    if money_match:
        earnings = float(money_match.group(1))
    harvest_match = re.search(r"harvest\s+(\d+)\s+crops?", text, re.IGNORECASE)
    if harvest_match:
        harvests = int(harvest_match.group(1))
    return earnings, harvests


def objective_plan_to_text():
    """The objective arithmetic: what is still missing and which crop closes it.

    The recorded `hard` episode compared its $2 cash balance with the $150 target,
    concluded "I cannot reach $150" and spent its remaining steps on `rest`, while
    the number actually being scored - `progress.earnings` - was still $0 because
    it had never called `sell`. The gap is therefore stated against the progress
    counter, together with the sales each crop needs and the cycles that still fit
    before the deadline.
    """
    progress = ((state or {}).get("objective") or {}).get("progress") or {}
    if "earnings" not in progress and "harvests" not in progress:
        return ""
    target_earnings, target_harvests = objective_targets()
    earned = progress.get("earnings") or 0
    harvests = progress.get("harvests") or 0
    crops = SERVER_CONFIG.get("crops") or {}
    lines = ["plan (arithmetic on the objective, not on your cash):"]

    if isinstance(target_earnings, float):
        gap = max(0.0, target_earnings - earned)
        lines.append(f"  earnings: {earned:g} of ${target_earnings:g} -> ${gap:.2f} still to sell."
                     " Your wallet does NOT count and neither does produce sitting in the"
                     " silo - only the `sell` action moves this number")
        priced = [((crop.get("yield_kg") or 0) * (crop.get("sell_price") or 0), name)
                  for name, crop in crops.items()
                  if (crop.get("yield_kg") or 0) * (crop.get("sell_price") or 0) > 0]
        if priced and gap > 0:
            priced.sort(reverse=True)
            best_gross, best_name = priced[0]
            worst_gross, worst_name = priced[-1]
            needed = math.ceil(gap / best_gross - 1e-9)
            lines.append(f"  sales: {best_name} is the money crop at ~${best_gross:.0f}/harvest ->"
                         f" {needed} sale(s) of it close the gap ({worst_name} only"
                         f" ~${worst_gross:.0f}/harvest)")

    if isinstance(target_harvests, int):
        left = max(0, target_harvests - harvests)
        lines.append(f"  harvests: {harvests} of {target_harvests} -> {left} more crop(s) to bring in."
                     " Each harvest needs its own plot, or the same plot on a second cycle")

    remaining = ticks_until_deadline()
    if remaining is not None and crops:
        days_left = remaining * MINUTES_PER_TICK / MINUTES_PER_DAY
        cycles = [f"{name} {int(days_left // crop['days'])}"
                  for name, crop in crops.items()
                  if isinstance(crop.get("days"), (int, float)) and crop["days"] > 0]
        if cycles:
            lines.append(f"  clock: {days_left:.1f} day(s) to the deadline; whole crop cycles that"
                         f" still fit: {', '.join(cycles)}. One cycle is plant -> harvest, so a plot"
                         " yields a second harvest only after another full cycle")
    return "\n".join(lines)


def crop_days_span():
    """Ripening span across the served catalogue, e.g. ``"3-5"`` (days)."""
    days = [crop.get("days") for crop in (SERVER_CONFIG.get("crops") or {}).values()
            if isinstance(crop.get("days"), (int, float))]
    if not days:
        return "3-5"
    lo, hi = min(days), max(days)
    return f"{lo:g}-{hi:g}" if lo != hi else f"{hi:g}"


# ---------------------------------------------------------------------------
# System prompt (one prompt, exactly like the legacy baseline)
# ---------------------------------------------------------------------------
def build_system_message(current, schemas, task):
    catalogue = "\n".join(f"- {entry['function']['name']}: {entry['function']['description']}"
                          for entry in schemas)
    landmarks = ", ".join(f"{name} ({pos[0]},{pos[1]})"
                          for name, pos in current["landmarks"].items())
    world = current["world"]
    objective = current["objective"]
    # Never let the prompt ask for nothing: the task block used to be blank, which
    # left the model free to ignore the objective and wander off.
    task = (task or "").strip() or objective["text"]
    status = ("complete" if objective["is_complete"]
              else "failed" if objective["is_failed"] else "open")
    # Rovers and water budget are stated because the recorded baseline lost the
    # game by hand-watering (one step per plot per day) and by never dispatching:
    # `robot_tiles` stayed 0 for three whole episodes.
    rovers = current.get("robots") or []
    rovers = ", ".join(robot["id"] for robot in rovers) or "none"
    litres = litres_per_watering()
    can_litres = current["player"]["can"]["capacity"]
    can_text = f"watering can {can_litres} L"
    if litres:
        can_text += (f" ({litres:g} L per `water`, about {can_litres / litres:.0f}"
                     f" waterings per fill)")
    silo = current["landmarks"].get("silo")
    wallet_text = starting_wallet_text()
    season_days = days_per_season()
    season_note = (f"Seasons last {season_days:g} day(s) each (spring, summer, autumn, winter)"
                   if isinstance(season_days, (int, float)) else "Seasons turn in-game")
    plan_text = objective_plan_to_text()
    return f"""
        you are a farm agent controlling one farmer in a server-authoritative farm game.

        Your task:
        {task}

        IMPORTANT.
        - The task above is the objective the server scores. Keep calling one tool after
        another until the state reports `objective ...` complete; the episode ends when
        the objective completes or misses its deadline. Do not idle or explore for its
        own sake while the objective is open.
        - One tool call does one action; the server is the only source of truth, so read the
        state that comes back with every result and never assume an action worked.
        - Tiles are [x, y] with x in 0..{world['w'] - 1} and y in 0..{world['h'] - 1}.
        Use `move_to` to travel and `move` to adjust by one tile.
        - `till`, `plant_crop`, `water`, `harvest`, `clear`, `weed`, `treat` and `fertilize`
        work on a plot within 1 tile of the farmer (standing on the plot is fine).
        - `collect_water` needs the farmer next to the tank, `buy` and `sell` need to be
        within 2 tiles of the silo.
        - A crop only grows while its soil is moist, and a crop left bone dry dies after
        roughly four days: read `needs_water` / `moisture` instead of watering on a fixed
        schedule, and refill the can at the tank when it runs low.
        - Rovers do the watering. Yours: {rovers}. One `dispatch` call per rover with task
        "water" puts it on irrigation duty for that single step: it drives to the tank to
        refill, waters the dry crop plots on its own, and returns to the charger when its
        battery drops. Watering by hand costs one step per plot per day, which is most of
        a small step budget - delegate the watering and spend your own steps on tilling,
        planting, harvesting and selling.
        - Harvested crops go into the silo, and nothing leaves it by itself. Sell before you
        spend: the cash for the next round of seed is the cash from the last sale, so as soon
        as a harvest lands in the silo `move_to` a tile within 2 of the silo
        ({silo[0]},{silo[1]}) and `sell` it - before tilling, planting or buying anything else.
        The money half of the objective counts `sell` proceeds, not the balance in your
        wallet, so produce left in the silo earns nothing.
        - Your starting wallet is {wallet_text}. Seed costs $4-7 each, so read `money` in the
        snapshot before a `buy`: a purchase you cannot afford answers NO_MONEY and costs you
        the step. Plant the seed you already hold first - it is already paid for - then buy
        into the crop with the best gross per harvest that the wallet still covers.
        - The snapshot can leave out plots that are far from the farmer (fog of war on the
        harder presets): `scan` the nearby plots or `inspect` one tile to read it anyway.
        - Weeds and disease slow growth and shrink the harvest: `weed` the plot (2 energy)
        and `treat` it ($4, 2 energy) when the crop shows weeds or disease.
        - Crops take {crop_days_span()} in-game days to ripen (per-crop table below). Use
        `advance` (ticks, 1 tick = {MINUTES_PER_TICK:g} minutes, {MINUTES_PER_DAY:g}
        minutes = 1 day) instead of waiting, then check the plot again. One call moves at
        most a day of farm time, never runs past the objective deadline and costs one of
        your steps, so a 3-day wait is 3 calls.
        - {season_note}, and the first line of every snapshot names the one you are standing
        in. A crop only grows in the seasons listed in its row above, and it grows slower as
        the temperature leaves its comfort band (winter is the cold one). Read the `day` and
        `season`, then place every harvest you still need before your crop drops out of
        season - a crop that is still growing when the season turns does not ripen in time.
        - Harvests come from plots: a crop you did not plant cannot be harvested. Cover every
        plot your seeds, money and energy allow - till, plant and water a block of them in one
        trip - and re-plant a plot during the same visit in which you harvested it.
        - Every observation ends with a `budget:` line: your steps left and the in-game
        days left before the deadline. Water the plots before you advance and spend steps
        on the objective rather than on recovering from a wasted walk.
        - Managed actions spend energy; `rest` at the farmhouse refills it. A walk that
        cannot take a single step fails with NO_ENERGY instead of doing nothing.
        - A failed action answers status "false" plus a `reason` (TOO_FAR, NO_PLOT,
        NO_SEEDS, NOT_READY, ...). Read the reason and correct the plan instead of retrying
        the same call blindly.

        WORLD
        size {world['w']}x{world['h']} tiles, {current['tick_ms']} ms per tick, now {current['time']['clock']}
        landmarks: {landmarks}
        rocks block movement: {', '.join(f"({rock['tile'][0]},{rock['tile'][1]})" for rock in current['obstacles'])}
        {crop_economics_to_text()}
        {can_text}, silo {current['silo']['capacity_kg']} kg, rovers: {rovers}
        objective [{status}] {objective['id']}: {objective['text']}
        progress {json.dumps(objective['progress'])} | deadline day {objective['deadline_day']} | reward ${objective['reward']}
        {plan_text}

        CURRENT STATE (authoritative snapshot, refreshed after every action):
        {state_to_text(current)}

        Never output tool arguments as text, JSON, markdown or code blocks.
        When an action is required, invoke the corresponding tool.

        Tools available:
        {catalogue}
"""


def tool_arguments(tool_call):
    """`arguments` is a dict in ollama>=0.4 but may arrive as a JSON string."""
    arguments = getattr(tool_call.function, "arguments", None) or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            logger.error(f"tool_call arguments are not valid JSON: {arguments!r}")
            return {}
    return dict(arguments)


# ---------------------------------------------------------------------------
# Smoke test: the API/tool layer without an LLM
# ---------------------------------------------------------------------------
def find_plot(tile):
    for plot in state.get("plots", []):
        if plot["x"] == int(tile[0]) and plot["y"] == int(tile[1]):
            return plot
    return None


def run_check_step(steps, name, function_to_call, **arguments):
    """Call a wrapper, record it like the agent loop does and keep the reply."""
    started = time.time()
    reply = json.loads(function_to_call(**arguments))
    steps.append((name, reply))
    if reply.get("status") == "true":
        tool_calls.append(name)
        time_taken.append(time.time() - started)
    return reply


def refill_can(steps, liters_needed):
    """Walk to the tank and fill the can when it holds less than `liters_needed`."""
    if state["player"]["can"]["litres"] >= liters_needed:
        return
    run_check_step(steps, "move_to", move_to, tile=state["landmarks"]["tank"])
    run_check_step(steps, "collect_water", collect_water)


def rest_at_farmhouse(steps):
    run_check_step(steps, "move_to", move_to, tile=state["landmarks"]["farmhouse"])
    run_check_step(steps, "rest", rest)


def plot_view(steps, tile):
    """Snapshot entry for `tile`, refreshed with `inspect` when fog hides it.

    The `hard` preset has fog of war: plots farther than 10 tiles from the
    farmer come back as `known: false`, while `inspect` always answers.
    """
    plot = find_plot(tile)
    if plot and plot.get("known", True):
        return plot
    reply = run_check_step(steps, "inspect", inspect, tile=tile)
    return (reply.get("data") or {}).get("plot") or plot


def choose_target_plot():
    """Nearest plot the scripted run can drive, plus whether it needs tilling.

    A fresh episode has empty, untilled plots; after `--no-reset` on a used
    episode the leftovers are tilled and empty (a harvest clears the crop but
    keeps the tilth), so `till` is skipped instead of failing.
    """
    known = sorted((plot for plot in state.get("plots", []) if plot.get("known", True)),
                   key=lambda plot: (plot.get("distance", 0), plot["id"]))
    untilled = [plot for plot in known if not plot.get("crop") and not plot.get("tilled")]
    if untilled:
        return untilled[0], True
    tilled = [plot for plot in known if not plot.get("crop") and plot.get("tilled")]
    if tilled:
        return tilled[0], False
    return None, False


def run_scripted_check(args):
    """Fixed till -> plant -> water -> advance -> harvest -> sell run."""
    global stop_reason
    print("=== check-api: driving the farm-sim API without an LLM ===")
    prepare_episode(args)

    plot, needs_till = choose_target_plot()
    if plot is None:
        print("check-api: no empty plot left in this episode - run without --no-reset "
              "to start a fresh one")
        logger.error("check-api: no empty plot available")
        return 1
    plot_tile = [plot["x"], plot["y"]]
    print(f"target plot {plot_tile}{'' if needs_till else ' (already tilled)'}")

    steps = []
    run_check_step(steps, "move_to", move_to, tile=plot_tile)
    if needs_till:
        run_check_step(steps, "till", till, tile=plot_tile)
    run_check_step(steps, "plant_crop", plant_crop, crop="wheat", tile=plot_tile)

    refill_can(steps, 2)
    run_check_step(steps, "move_to", move_to, tile=plot_tile)
    run_check_step(steps, "water", water, tile=plot_tile)

    plot = plot_view(steps, plot_tile)
    for _ in range(30):
        if plot and (plot.get("crop") or {}).get("ready"):
            break
        # Travelling and watering cost energy, so top up before doing work.
        if state["player"]["energy"] < 30:
            rest_at_farmhouse(steps)
            run_check_step(steps, "move_to", move_to, tile=plot_tile)
        if plot and (plot.get("crop") or {}).get("weeds"):
            run_check_step(steps, "weed", weed, tile=plot_tile)
        if plot and plot.get("needs_water"):
            refill_can(steps, 2)
            if [state["player"]["tile"][0], state["player"]["tile"][1]] != plot_tile:
                run_check_step(steps, "move_to", move_to, tile=plot_tile)
            run_check_step(steps, "water", water, tile=plot_tile)
        run_check_step(steps, "advance", advance, ticks=60)
        plot = plot_view(steps, plot_tile)
    print(f"plot after advancing: {describe_plot(plot) if plot else 'unknown'}")

    replies = {}
    run_check_step(steps, "harvest", harvest, tile=plot_tile)
    run_check_step(steps, "move_to", move_to, tile=state["landmarks"]["silo"])
    run_check_step(steps, "sell", sell, crop="wheat")
    required = ["plant_crop", "harvest", "sell"] + (["till"] if needs_till else [])
    for name in required:
        replies[name] = next(reply for step_name, reply in reversed(steps) if step_name == name)

    print("\n=== check-api results ===")
    for name, reply in steps:
        print(f"  {name:<14} status={reply.get('status', '?'):<5} {reply.get('message', '')}")
    metrics = state["metrics"]
    print(f"  money={state['player']['money']} kg_sold={metrics['kg_sold']} "
          f"earned={metrics['earned']} invalid_actions={metrics['invalid_actions']}")

    ok = all(replies[name].get("status") == "true" for name in required) and metrics["kg_sold"] > 0
    stop_reason = "check_api_pass" if ok else "check_api_fail"
    print("check-api: PASS" if ok else "check-api: FAIL")
    logger.info(f"check-api finished ok={ok} metrics={json.dumps(metrics)}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# The agentic loop - identical shape to the legacy baseline
# ---------------------------------------------------------------------------
def run_agent(args, schemas):
    global invalid_moves, total_input_tokens, total_output_tokens, stop_reason
    global STEPS_USED

    # The prompt and the first user turn both carry the task; `--task` overrides
    # it, otherwise it is the objective this preset assigns (never empty, which is
    # what used to leave the model without a task to obey).
    task_text = resolve_task(args)
    system_message2 = build_system_message(state, schemas, task_text)
    messages = [
        {'role': 'system', 'content': system_message2},
        {'role': 'user', 'content': task_text},
    ]
    print(f"task: {task_text}")
    logger.info(f"task: {task_text}")
    logger.info(f"system message: {system_message2}")

    client = Client(host=config_data["server_urls"]["ollama_url"], timeout=args.llm_timeout)

    silent_steps = 0  # consecutive model turns without a tool call
    stuck_repeats = 0  # consecutive identical tool calls
    last_signature = None
    try:
        i = 0
        while i < args.max_steps:
            i += 1
            # Kept at module level so `budget_to_text` can report what is left in the
            # observation the model reads after this step's action.
            STEPS_USED = i
            st_time = time.time()
            response: ChatResponse = client.chat(model=args.model, messages=messages, tools=schemas)

            print(f"input_tokens: {response['prompt_eval_count']}")
            print("Prompt evaluation time:", response["prompt_eval_duration"] / 1e9, "seconds")
            logger.info(f"Prompt evaluation time:{response['prompt_eval_duration'] / 1e9:.2f}")
            print("Generation time:", response["eval_duration"] / 1e9, "seconds")
            logger.info(f"Generation time: {response['eval_duration'] / 1e9:.2f}")
            print(f"output_tokens: {response['eval_count']}")
            print(f"response time: {(response['total_duration'] / 1e9)}")
            logger.info(f"step {i} response time: {(response['total_duration'] / 1e9):.2f}s")

            total_output_tokens += response['eval_count'] or 0
            total_input_tokens += response['prompt_eval_count'] or 0

            print("response:")
            print(response.message)
            if response.message.content:
                print('Content: ')
                print(response.message.content + '\n')
                logger.info(f"content : {response.message.content}")

            if getattr(response.message, "thinking", None):
                print('Thinking: ')
                print(response.message.thinking + '\n')
                logger.info(f"thinking : {response.message.thinking}")

            messages.append(response.message)

            if response.message.tool_calls:
                silent_steps = 0
                print("llm tool_calls:")
                print(len(response.message.tool_calls))
                for tool_call in response.message.tool_calls:
                    # The LLM decides which wrapper to call, we never repair its plan.
                    name = tool_call.function.name
                    arguments = tool_arguments(tool_call)
                    # A model that repeats one call forever burns the whole step
                    # budget: remember the call so the loop can nudge it once.
                    signature = (name, json.dumps(arguments, sort_keys=True, default=str))
                    if signature == last_signature:
                        stuck_repeats += 1
                    else:
                        last_signature, stuck_repeats = signature, 1
                    function_to_call = available_tools.get(name)
                    logger.info(f"tool call: {name} args={json.dumps(arguments)}")
                    if function_to_call:
                        try:
                            real_result_json = function_to_call(**arguments)
                        except TypeError as e:
                            invalid_moves += 1
                            key = "bad tool arguments"
                            invalid_move_object[key] = invalid_move_object.get(key, 0) + 1
                            real_result_json = json.dumps({"status": "false", "action": name,
                                                           "message": f"Error in tool call: {str(e)}"})
                    else:
                        invalid_moves += 1
                        key = "unknown tool"
                        invalid_move_object[key] = invalid_move_object.get(key, 0) + 1
                        print(f'Tool {name} not found')
                        messages.append({'role': 'tool', 'content': f'Tool {name} not found',
                                         'tool_name': name})
                        continue

                    print(f"time for tool {name}: {str(time.time() - st_time)}")
                    if json.loads(real_result_json).get("status") == "true":
                        tool_calls.append(name)
                        time_taken.append(time.time() - st_time)
                    messages.append({
                        "role": "tool",
                        "content": json.dumps({
                            "action_result": json.loads(real_result_json),
                            "current_state": state_to_text(state),
                        }),
                        "tool_name": name,
                    })

                if stuck_repeats >= STUCK_REPEATS:
                    # Naming the repeat is the cheapest way out of a loop: the model
                    # sees the same arguments it just sent and a list of alternatives.
                    print(f"stuck: {last_signature[0]} called {stuck_repeats}x with the same arguments")
                    logger.info(f"stuck: {last_signature[0]} called {stuck_repeats}x with the same "
                                f"arguments {last_signature[1]} - hinting the model")
                    messages.append({
                        'role': 'user',
                        'content': (f"You have called `{last_signature[0]}` with identical arguments "
                                    f"{stuck_repeats} times and the state did not change. Read "
                                    "`reason` and `note` in the last action_result and change the plan: "
                                    "`rest` when energy is low, `move` one tile when the path is "
                                    "blocked, `inspect` the tile when you are unsure."),
                    })
                    last_signature, stuck_repeats = None, 0
            elif state.get("objective", {}).get("is_complete"):
                print("goal completed")
                logger.info("objective completed")
                stop_reason = "objective_complete"
                break
            else:
                # A turn without a tool call used to end the episode, which reads as
                # "the agent ignored the task". Nudge it back, then give up.
                silent_steps += 1
                print(f"LLM did not call tools but goal is not complete (nudge {silent_steps}/2).")
                logger.info(f"LLM did not call tools but goal is not complete (nudge {silent_steps}/2)")
                if silent_steps >= 2:
                    logger.info("no tool call after 2 nudges - ending the episode")
                    stop_reason = "silent_steps"
                    break
                messages.append({
                    'role': 'user',
                    'content': ("You did not call a tool and the objective is still open: "
                                f"{task_text} Call exactly one tool now."),
                })

            if state.get("objective", {}).get("is_complete") or state.get("objective", {}).get("is_failed"):
                stop_reason = ("objective_complete"
                               if state["objective"].get("is_complete") else "objective_failed")
                print("objective finished - ending the episode")
                logger.info(f"objective finished: {state['objective']['id']} "
                            f"complete={state['objective']['is_complete']} "
                            f"failed={state['objective']['is_failed']}")
                break
        if not stop_reason:
            # The loop ran out of steps with the goal still open: that is a result,
            # not a silent end - the summary has to say so.
            stop_reason = "max_steps"
            print(f"stopped at max_steps ({args.max_steps}) - the objective is still open")
            logger.info(f"stopped at max_steps ({args.max_steps}) "
                        f"objective={json.dumps(state.get('objective'))}")
    except Exception as e:
        stop_reason = "llm_error"
        logger.error(f"LLm failed due to error: {str(e)}")
        print(f"llm failed due to error: {e}")
        # The usual cause: the Ollama host in config.yaml is unreachable, so no
        # tool call was ever sent and the game looks untouched.
        print(f"model server: {(config_data.get('server_urls') or {}).get('ollama_url')} "
              f"(config.yaml server_urls.ollama_url) - is it up and is '{args.model}' pulled?")
        print("nothing was posted to the game after the last tool call above")
        logger.error(f"model server: {(config_data.get('server_urls') or {}).get('ollama_url')}")


# ---------------------------------------------------------------------------
# Reporting - same statistics and log lines as the legacy baseline
# ---------------------------------------------------------------------------
def check_sequence_details(actual, correct):
    matched_elements = []

    for i, action in enumerate(actual):
        # Case 1: Actual sequence is longer than the correct sequence
        if i >= len(correct):
            print(f"Result: WRONG")
            print(f" -> Matched until: {matched_elements}")
            print(f" -> Reason: 'actual' sequence is longer than 'correct' sequence.")
            return False, matched_elements

        # Case 2: Element matches
        if action == correct[i]:
            matched_elements.append(action)
        else:
            # Case 3: Element does not match
            print(f"Result: WRONG")
            print(f" -> Matched until: {matched_elements}")
            print(f" -> At position {i}, expected '{correct[i]}' but got '{action}'")
            return False, matched_elements

    # If it finishes the loop, it's a perfect prefix match
    print(f"Result: CORRECT")
    print(f" -> Matched until: {matched_elements}")
    return True, matched_elements


def dedupe(sequence):
    """Collapse consecutive repetitions (`till, till, water` -> `till, water`)."""
    collapsed = []
    for action in sequence:
        if not collapsed or collapsed[-1] != action:
            collapsed.append(action)
    return collapsed


def print_step_order(sequence):
    """Position of the first occurrence of each canonical step.

    A strict prefix match (see `check_sequence_details`) is too brittle once the
    agent works several plots, so the order of the first occurrences is reported
    as well.
    """
    first_seen = {}
    for index, name in enumerate(sequence):
        first_seen.setdefault(name, index)
    print("step order (first occurrence):")
    for name in correct_seq:
        print(f"  {name:<14} {first_seen.get(name, '-')}")
    present = [first_seen[name] for name in correct_seq if name in first_seen]
    ordered = all(earlier < later for earlier, later in zip(present, present[1:]))
    print(f"  canonical steps present and in order: {ordered}")
    logger.info(f"step order: {json.dumps(first_seen)} ordered={ordered}")
    return first_seen, ordered


def summarise_run(args, stem):
    """Collect the episode metrics, write the latency plot and the JSON summary."""
    global state, invalid_moves
    refresh_state()

    metrics = state["metrics"]
    objective = state["objective"]
    points_gained_object = {
        "objective_completed": int(metrics.get("objective_completed", 0)),
        "harvests": int(metrics.get("harvests", 0)),
        "kg_harvested": metrics.get("kg_harvested", 0),
        "kg_sold": metrics.get("kg_sold", 0),
        "earned": metrics.get("earned", 0),
        "spent": metrics.get("spent", 0),
        "crops_died": int(metrics.get("crops_died", 0)),
    }
    # The legacy baseline counted these by hand; the simulation already scores
    # the episode, so the same log lines now carry the sim values.
    points_gained = metrics.get("score", 0)

    stats = {}
    tool_calls_final = []
    step_order = {}
    invalid_move_rate = 0.0

    print(time_taken)
    if len(time_taken) > 0 and len(tool_calls) > 0:
        data = time_taken
        stats = {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "variance": float(np.var(data)),
            "std_dev": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
        }

        print("No of llms calls:-")
        print(len(time_taken))
        logger.info(f"No of llms calls: {len(time_taken)}")
        logger.info(f"time taken stats: {json.dumps(stats)}")

        print("player positions:")
        print(player_positions)
        logger.info(f"player_positions: {player_positions}")

        print(f"points gained by agent: {str(points_gained)}")
        logger.info(f"points gained by agent: {str(points_gained)}")
        print(f"points gained object: {str(points_gained_object)}")
        logger.info(f"points gained object: {str(points_gained_object)}")

        logger.info(f"time taken values: {time_taken}")
        print("total input tokens: " + str(total_input_tokens))
        print("total output tokens: " + str(total_output_tokens))
        logger.info("total input tokens: " + str(total_input_tokens))
        logger.info("total output tokens: " + str(total_output_tokens))

        plt.figure()
        plt.plot(time_taken)
        plt.xlabel("successful tool call")
        plt.ylabel("seconds")
        plt.title(f"{args.model or 'check_api'} - farm-sim {args.preset} seed {args.seed}")
        plot_path = os.path.join(args.log_folder, f"{stem}_latency.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"latency plot: {plot_path}")
        logger.info(f"latency plot: {plot_path}")

        tool_calls_final = dedupe(tool_calls)
        print("--- Test sequence ---")
        check_sequence_details(tool_calls_final, correct_seq)
        print("sequence of tool calls:" + str(tool_calls_final))
        logger.info("sequence of tool calls:" + str(tool_calls_final))
        step_order, _ = print_step_order(tool_calls_final)

        invalid_moves += jumps
        invalid_move_rate = invalid_moves / max(1, len(player_positions))
        print("invalid move rate:")
        print(invalid_move_rate)
        print("no of invalid moves:" + str(invalid_moves))
        print("invalid move object:")
        print(invalid_move_object)
        logger.info("no of invalid moves:" + str(invalid_moves))
        logger.info("invalid move object:" + str(invalid_move_object))
        logger.info(f"invalid move rate: {invalid_move_rate}")
        logger.info("game character jumps:" + str(jumps))
        print("no of revisits:")
        print(str(revisits))
        logger.info("no of revisits:" + str(revisits))
    else:
        print(f"points gained by agent: {str(points_gained)}")
        logger.info(f"points gained by agent: {str(points_gained)}")
        logger.info("llm tool failed")
        print("llm tool failed")

    print("final state:")
    print(state_to_text(state))
    logger.info("final state:\n" + state_to_text(state))
    print(f"stop reason: {stop_reason or 'unknown'}")
    logger.info(f"stop reason: {stop_reason or 'unknown'}")

    summary = {
        "model": args.model,
        "base_url": BASE_URL,
        "preset": state["preset"],
        "objective_requested": args.objective or "preset default",
        "seed": state["seed"],
        "task": state["task"],
        "days_played": state["time"]["day"],
        "tick": state["tick"],
        "max_steps": args.max_steps,
        "stop_reason": stop_reason,
        "max_advance_ticks": MAX_ADVANCE_TICKS,
        "objective": objective,
        "metrics": metrics,
        "points_gained": points_gained,
        "points_gained_object": points_gained_object,
        "no_of_llm_calls": len(time_taken),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_calls": tool_calls,
        "tool_calls_deduped": tool_calls_final,
        "step_order": step_order,
        "time_taken": time_taken,
        "time_taken_stats": stats,
        "invalid_moves": invalid_moves,
        "invalid_move_object": invalid_move_object,
        "invalid_move_rate": invalid_move_rate,
        "jumps": jumps,
        "revisits": revisits,
        "player_positions": player_positions,
    }
    summary_path = os.path.join(args.log_folder, f"{stem}_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)
    print(f"summary: {summary_path}")
    logger.info(f"summary written to {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def describe_server(health):
    """One line that proves *which* server this run is talking to."""
    return (f"server: {health.get('version', '?')} preset={health.get('preset')} "
            f"seed={health.get('seed')} episode={health.get('episode')} "
            f"tick={health.get('tick')} browser clients={health.get('clients', 0)}")


def main(argv=None):
    global BASE_URL, MAX_ADVANCE_TICKS, MAX_STEPS
    args = parse_args(argv)
    BASE_URL = resolve_base_url(args)
    MAX_ADVANCE_TICKS = args.max_advance_ticks
    ollama_url = (config_data.get("server_urls") or {}).get("ollama_url")

    print(f"farm-sim: {BASE_URL}")
    print(f"model: {args.model or '(none - check-api)'} @ {ollama_url}")

    # Fail fast: no log file or episode reset happens when the server is down.
    try:
        health = api_get("/api/v1/health")
    except Exception as exc:
        print(f"cannot reach the farm-sim server at {BASE_URL}: {exc}")
        print('start it with: cd "game/phaser_test/farm_sim" && npm start')
        return 1

    # The health payload is the ground truth for which world this run drives:
    # a server started with a different PORT/PRESET is a different game.
    print(describe_server(health))
    print(f"open {BASE_URL} in the browser to watch this episode")
    if not health.get("clients"):
        print(f"note: no browser is connected to {BASE_URL} yet")

    if args.list_objectives:
        print_objectives()
        return 0

    # Read the catalogue before anything is reset: the server replaces an unknown
    # objective id with `starter_easy`, so a typo would otherwise score something
    # else entirely. This also names the run's preset/objective mix up front.
    load_objective_catalogue()
    exit_code = validate_objective(args)
    if exit_code:
        return exit_code

    # The served crop economics and preset constants, so the prompt quotes the
    # numbers this server actually scores instead of stale literals.
    load_server_config()

    stem, log_path = setup_logging(args)
    print(f"log file: {log_path}")
    logger.info(f"farm-sim base url: {BASE_URL}")
    logger.info(describe_server(health))
    logger.info(f"Running model: {args.model}")
    logger.info(f"args: {vars(args)}")
    # Recorded in the log, not just in the prompt: a server restarted with other
    # constants makes every number below stale, and this is where that shows.
    crops = SERVER_CONFIG.get("crops") or {}
    print(f"served config: {len(crops)} crops, "
          f"{SERVER_CONFIG.get('minutes_per_tick', '?')} min/tick, "
          f"water_per_action={litres_per_watering()}")
    logger.info(f"served config keys: {sorted(SERVER_CONFIG)}")
    logger.info(f"served crops: {json.dumps(crops, sort_keys=True)}")
    print(describe_objective(args))
    logger.info(describe_objective(args))
    print(f"advance cap: {MAX_ADVANCE_TICKS or 'off'} tick(s) per call, deadline-aware")
    logger.info(f"advance cap: {MAX_ADVANCE_TICKS} tick(s) per call (0 = no cap)")
    # The same number the loop enforces, published in every observation so the model
    # can budget its steps instead of discovering the limit by being stopped.
    MAX_STEPS = args.max_steps
    print(f"step budget: {MAX_STEPS} LLM step(s), reported in every observation")
    logger.info(f"step budget: {MAX_STEPS} LLM step(s), reported in every observation")

    if args.check_api:
        exit_code = run_scripted_check(args)
        summarise_run(args, stem)
        return exit_code

    prepare_episode(args)
    schemas = fetch_tool_schemas()
    print(f"tools from the server: {[entry['function']['name'] for entry in schemas]}")
    logger.info(f"tools from the server: {[entry['function']['name'] for entry in schemas]}")
    print(f"initial state:\n{state_to_text(state)}")
    logger.info(f"initial state:\n{state_to_text(state)}")

    started = time.time()
    run_agent(args, schemas)
    summarise_run(args, stem)

    print(f"episode finished in {time.time() - started:.1f}s")
    logger.info(f"episode finished in {time.time() - started:.1f}s")

    if args.reset_after:
        prepare_episode(args)
        print("episode reset after the run")
    return 0


if __name__ == "__main__":
    sys.exit(main())
