"""The system prompt and tool-argument decoding.

One prompt, exactly like the legacy baseline. Its text is the experiment, so the
wording, the bullet order and the whitespace stay as they were in the single-file
script; only the source of the numbers changed (everything that can be served by
the sim is quoted from `farm_agent.served`).
"""

import json
import logging

from . import runtime
from .rendering import state_to_text
from .served import (crop_days_span, crop_economics_to_text, days_per_season,
                     litres_per_watering, objective_plan_to_text, starting_wallet_text)

logger = logging.getLogger(__name__)


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
    minutes_per_tick = runtime.MINUTES_PER_TICK
    minutes_per_day = runtime.MINUTES_PER_DAY
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
        `advance` (ticks, 1 tick = {minutes_per_tick:g} minutes, {minutes_per_day:g}
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
