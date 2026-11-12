"""Reporting - the same statistics and log lines as the legacy baseline."""

import json
import logging
import os

import matplotlib
matplotlib.use("Agg")  # experiment runs are headless
import matplotlib.pyplot as plt
import numpy as np

from . import runtime
from .api import refresh_state
from .rendering import state_to_text

logger = logging.getLogger(__name__)

#: Tool sequence a competent baseline run produces at least once.
CORRECT_SEQ = ['till', 'plant_crop', 'collect_water', 'water', 'harvest', 'sell']


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
    for name in CORRECT_SEQ:
        print(f"  {name:<14} {first_seen.get(name, '-')}")
    present = [first_seen[name] for name in CORRECT_SEQ if name in first_seen]
    ordered = all(earlier < later for earlier, later in zip(present, present[1:]))
    print(f"  canonical steps present and in order: {ordered}")
    logger.info(f"step order: {json.dumps(first_seen)} ordered={ordered}")
    return first_seen, ordered


def summarise_run(args, stem):
    """Collect the episode metrics, write the latency plot and the JSON summary."""
    refresh_state()

    metrics = runtime.state["metrics"]
    objective = runtime.state["objective"]
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

    print(runtime.time_taken)
    if len(runtime.time_taken) > 0 and len(runtime.tool_calls) > 0:
        data = runtime.time_taken
        stats = {
            "mean": float(np.mean(data)),
            "median": float(np.median(data)),
            "variance": float(np.var(data)),
            "std_dev": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
        }

        print("No of llms calls:-")
        print(len(runtime.time_taken))
        logger.info(f"No of llms calls: {len(runtime.time_taken)}")
        logger.info(f"time taken stats: {json.dumps(stats)}")

        print("player positions:")
        print(runtime.player_positions)
        logger.info(f"player_positions: {runtime.player_positions}")

        print(f"points gained by agent: {str(points_gained)}")
        logger.info(f"points gained by agent: {str(points_gained)}")
        print(f"points gained object: {str(points_gained_object)}")
        logger.info(f"points gained object: {str(points_gained_object)}")

        logger.info(f"time taken values: {runtime.time_taken}")
        print("total input tokens: " + str(runtime.total_input_tokens))
        print("total output tokens: " + str(runtime.total_output_tokens))
        logger.info("total input tokens: " + str(runtime.total_input_tokens))
        logger.info("total output tokens: " + str(runtime.total_output_tokens))

        plt.figure()
        plt.plot(runtime.time_taken)
        plt.xlabel("successful tool call")
        plt.ylabel("seconds")
        plt.title(f"{args.model or 'check_api'} - farm-sim {args.preset} seed {args.seed}")
        plot_path = os.path.join(args.log_folder, f"{stem}_latency.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"latency plot: {plot_path}")
        logger.info(f"latency plot: {plot_path}")

        tool_calls_final = dedupe(runtime.tool_calls)
        print("--- Test sequence ---")
        check_sequence_details(tool_calls_final, CORRECT_SEQ)
        print("sequence of tool calls:" + str(tool_calls_final))
        logger.info("sequence of tool calls:" + str(tool_calls_final))
        step_order, _ = print_step_order(tool_calls_final)

        runtime.invalid_moves += runtime.jumps
        invalid_move_rate = runtime.invalid_moves / max(1, len(runtime.player_positions))
        print("invalid move rate:")
        print(invalid_move_rate)
        print("no of invalid moves:" + str(runtime.invalid_moves))
        print("invalid move object:")
        print(runtime.invalid_move_object)
        logger.info("no of invalid moves:" + str(runtime.invalid_moves))
        logger.info("invalid move object:" + str(runtime.invalid_move_object))
        logger.info(f"invalid move rate: {invalid_move_rate}")
        logger.info("game character jumps:" + str(runtime.jumps))
        print("no of revisits:")
        print(str(runtime.revisits))
        logger.info("no of revisits:" + str(runtime.revisits))
    else:
        print(f"points gained by agent: {str(points_gained)}")
        logger.info(f"points gained by agent: {str(points_gained)}")
        logger.info("llm tool failed")
        print("llm tool failed")

    print("final state:")
    print(state_to_text(runtime.state))
    logger.info("final state:\n" + state_to_text(runtime.state))
    print(f"stop reason: {runtime.stop_reason or 'unknown'}")
    logger.info(f"stop reason: {runtime.stop_reason or 'unknown'}")

    summary = {
        "model": args.model,
        "base_url": runtime.base_url,
        "preset": runtime.state["preset"],
        "objective_requested": args.objective or "preset default",
        "seed": runtime.state["seed"],
        "task": runtime.state["task"],
        "days_played": runtime.state["time"]["day"],
        "tick": runtime.state["tick"],
        "max_steps": args.max_steps,
        "stop_reason": runtime.stop_reason,
        "max_advance_ticks": runtime.MAX_ADVANCE_TICKS,
        "objective": objective,
        "metrics": metrics,
        "points_gained": points_gained,
        "points_gained_object": points_gained_object,
        "no_of_llm_calls": len(runtime.time_taken),
        "total_input_tokens": runtime.total_input_tokens,
        "total_output_tokens": runtime.total_output_tokens,
        "tool_calls": runtime.tool_calls,
        "tool_calls_deduped": tool_calls_final,
        "step_order": step_order,
        "time_taken": runtime.time_taken,
        "time_taken_stats": stats,
        "invalid_moves": runtime.invalid_moves,
        "invalid_move_object": runtime.invalid_move_object,
        "invalid_move_rate": invalid_move_rate,
        "jumps": runtime.jumps,
        "revisits": runtime.revisits,
        "player_positions": runtime.player_positions,
    }
    summary_path = os.path.join(args.log_folder, f"{stem}_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as file:
        json.dump(summary, file, indent=2)
    print(f"summary: {summary_path}")
    logger.info(f"summary written to {summary_path}")
    return summary
