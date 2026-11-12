"""Smoke test: the API/tool layer without an LLM.

`--check-api` drives a fixed till -> plant -> water -> advance -> harvest -> sell
run through the same wrappers, the same `call_action` bookkeeping and the same
summary as a model run, so a broken server or a broken tool wrapper shows up
before an experiment is spent on it.
"""

import json
import logging
import time

from . import runtime
from .episode import prepare_episode
from .rendering import describe_plot
from .tools import (advance, collect_water, harvest, inspect, move_to, plant_crop,
                    rest, sell, till, water, weed)

logger = logging.getLogger(__name__)


def find_plot(tile):
    for plot in runtime.state.get("plots", []):
        if plot["x"] == int(tile[0]) and plot["y"] == int(tile[1]):
            return plot
    return None


def run_check_step(steps, name, function_to_call, **arguments):
    """Call a wrapper, record it like the agent loop does and keep the reply."""
    started = time.time()
    reply = json.loads(function_to_call(**arguments))
    steps.append((name, reply))
    if reply.get("status") == "true":
        runtime.tool_calls.append(name)
        runtime.time_taken.append(time.time() - started)
    return reply


def refill_can(steps, liters_needed):
    """Walk to the tank and fill the can when it holds less than `liters_needed`."""
    if runtime.state["player"]["can"]["litres"] >= liters_needed:
        return
    run_check_step(steps, "move_to", move_to, tile=runtime.state["landmarks"]["tank"])
    run_check_step(steps, "collect_water", collect_water)


def rest_at_farmhouse(steps):
    run_check_step(steps, "move_to", move_to, tile=runtime.state["landmarks"]["farmhouse"])
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
    known = sorted((plot for plot in runtime.state.get("plots", []) if plot.get("known", True)),
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
        if runtime.state["player"]["energy"] < 30:
            rest_at_farmhouse(steps)
            run_check_step(steps, "move_to", move_to, tile=plot_tile)
        if plot and (plot.get("crop") or {}).get("weeds"):
            run_check_step(steps, "weed", weed, tile=plot_tile)
        if plot and plot.get("needs_water"):
            refill_can(steps, 2)
            if ([runtime.state["player"]["tile"][0], runtime.state["player"]["tile"][1]]
                    != plot_tile):
                run_check_step(steps, "move_to", move_to, tile=plot_tile)
            run_check_step(steps, "water", water, tile=plot_tile)
        run_check_step(steps, "advance", advance, ticks=60)
        plot = plot_view(steps, plot_tile)
    print(f"plot after advancing: {describe_plot(plot) if plot else 'unknown'}")

    replies = {}
    run_check_step(steps, "harvest", harvest, tile=plot_tile)
    run_check_step(steps, "move_to", move_to, tile=runtime.state["landmarks"]["silo"])
    run_check_step(steps, "sell", sell, crop="wheat")
    required = ["plant_crop", "harvest", "sell"] + (["till"] if needs_till else [])
    for name in required:
        replies[name] = next(reply for step_name, reply in reversed(steps) if step_name == name)

    print("\n=== check-api results ===")
    for name, reply in steps:
        print(f"  {name:<14} status={reply.get('status', '?'):<5} {reply.get('message', '')}")
    metrics = runtime.state["metrics"]
    print(f"  money={runtime.state['player']['money']} kg_sold={metrics['kg_sold']} "
          f"earned={metrics['earned']} invalid_actions={metrics['invalid_actions']}")

    ok = all(replies[name].get("status") == "true" for name in required) and metrics["kg_sold"] > 0
    runtime.stop_reason = "check_api_pass" if ok else "check_api_fail"
    print("check-api: PASS" if ok else "check-api: FAIL")
    logger.info(f"check-api finished ok={ok} metrics={json.dumps(metrics)}")
    return 0 if ok else 1
