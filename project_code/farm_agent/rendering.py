"""The "CURRENT STATE" block the model reads.

Every observation handed to the model is this module's output, so the wording,
the ordering (budget before the map, map before the events) and the ASCII field
grid are part of the experiment: changing them changes what the model sees.
"""

import json

from . import budget

#: One letter per crop for the ASCII field grid.
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
    deadline, per-call advance cap) - see `farm_agent.budget.budget_to_text`.
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
    budget_line = budget.budget_to_text()
    if budget_line:
        lines.append(budget_line)

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
