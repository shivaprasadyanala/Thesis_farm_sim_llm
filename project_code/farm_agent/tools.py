"""Tools - every tool is one action on the server, nothing is simulated locally.

Each wrapper only names its action and its arguments; the server owns the rules
and answers with the fresh observation. The docstrings are the tool descriptions
the model reads, so they are written for the model, not for a developer.

One call is one `POST /api/v1/action`, and the wrapper turns the answer into the
JSON string the model reads. A turn may order several calls; they are executed in
that order, one request each, and each answer is read against the state that call
left behind. The model cannot react between the calls of its own turn (the
assistant message that ordered them is already written), so the harness never
merges, reorders or retries them: the model's plan is the plan.
"""

import json
import logging
import time

from . import runtime
from .api import api_get, api_post
from .budget import clamp_advance, requested_ticks
from .tracking import track_position

logger = logging.getLogger(__name__)

#: Directions `move` accepts; anything else is refused here, without a round trip.
MOVE_DIRECTIONS = ("up", "down", "left", "right")


def progressless(data):
    """True when an action claimed success but the farmer did not move at all.

    `move_to` answers `steps: 0` / `reached: false` when its walk loop stops on
    `energy < 1`. Counting that as a valid action hides the failure from the
    invalid-move table and leaves the model with no reason to `rest`, so the
    harness turns it into a failure with an actionable message.
    """
    return data.get("reached") is False and data.get("steps") == 0


def call_action(payload, note=None):
    """Run one action and return the reply as the JSON string the model reads.

    The reply keeps the shape the legacy tools used (`status` / `action` /
    `message`) so the same evaluation code can read both experiments, plus the
    server's machine-readable `reason` and `data`. A `note` (used by `advance` for
    a clipped jump) is merged into the reply *before* it is printed and logged, so
    the trace shows the same thing the model saw.
    """
    action = str(payload.get("type", "?"))
    # Drop unset optional arguments: the server treats `null` as a value
    # (`move_to` with steps=null would only walk one tile, `water` with
    # tile=null targets the faced tile), so absent is not the same as null.
    payload = {key: value for key, value in payload.items() if value is not None}
    started = time.time()
    result = send(payload, action)
    return json.dumps(apply_result(payload, result, note, time.time() - started))


def reject(action, reason, message):
    """Fail a call locally, without spending a round trip, and book-keep the refusal."""
    runtime.invalid_moves += 1
    runtime.invalid_move_object[reason] = runtime.invalid_move_object.get(reason, 0) + 1
    print(f"tool {action} -> ok=False reason={reason} message={message!r} (refused locally)")
    logger.info(f"tool: {action} refused locally: {reason} - {message}")
    return json.dumps({"status": "false", "action": action, "message": message,
                       "reason": reason, "data": {}})


def send(payload, action):
    """POST one action; an HTTP failure becomes a failed reply, never an exception."""
    try:
        return api_post("/api/v1/action", payload, include_state=True)
    except Exception as exc:
        logger.error(f"tool: {action} failed over HTTP: {exc}")
        return {"ok": False, "reason": "transport error",
                "message": f"transport error: {exc}", "data": {}}


def apply_result(payload, result, note, tool_time):
    """Build the reply for one server result and book-keep it.

    Every action ends up here, in call order, so `invalid_moves`, `tool_calls`,
    `time_taken`, `visited`/`revisits`/`jumps` and `player_positions` are booked in
    exactly one place. `tool_time` is how long that call's request took, booked
    here because this is the only layer that knows when the server answered (the
    loop used to book the whole LLM step as tool latency).
    """
    action = str(payload.get("type", "?"))
    if isinstance(result.get("state"), dict):
        runtime.state = result.pop("state")

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
        runtime.invalid_moves += 1
        runtime.invalid_move_object[reason] = runtime.invalid_move_object.get(reason, 0) + 1
    else:
        runtime.tool_calls.append(action)
        runtime.time_taken.append(tool_time)

    track_position(action)
    print(f"tool {action} -> ok={ok} reason={reply.get('reason')} "
          f"message={reply['message']!r} ({tool_time:.3f}s)")
    logger.info(f"tool: {action} args={json.dumps({k: v for k, v in payload.items() if k != 'type'})} "
                f"result: {json.dumps(reply)} tool_time={tool_time:.3f}s")
    return reply


def move(dir: str = None) -> str:
    """Move the farmer exactly one tile north/south/east/west.

    Args:
        dir: 'up', 'down', 'left' or 'right'.
    """
    if dir is not None and str(dir).strip().lower() not in MOVE_DIRECTIONS:
        # Refused here, not by the server: a typo used to cost a round trip and came
        # back as BAD_ARGS without naming the four directions the model may send.
        return reject("move", "BAD_ARGS",
                      f"Unknown direction {dir!r} - use one of {', '.join(MOVE_DIRECTIONS)}.")
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
    # Clamped against the clock the harness has just read: every call is applied as
    # it is made, so a second `advance` in the same turn sees the first one's result
    # and cannot push the world past the objective deadline.
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
    if runtime.MAX_ADVANCE_TICKS > 0:
        per_call = runtime.MAX_ADVANCE_TICKS * runtime.MINUTES_PER_TICK / runtime.MINUTES_PER_DAY
        for entry in schemas:
            if entry["function"]["name"] == "advance":
                entry["function"]["description"] = (
                    f"{entry['function']['description']} Capped: one call moves at most "
                    f"{runtime.MAX_ADVANCE_TICKS} ticks ({per_call:.0f} farm day), never past the "
                    "objective deadline, and costs one step - a 3-day wait is 3 calls, so "
                    "water dry plots before waiting.")
    missing = [entry["function"]["name"] for entry in schemas
               if entry["function"]["name"] not in available_tools]
    if missing:
        raise RuntimeError(f"server advertises tools without a local wrapper: {missing}")
    return schemas
