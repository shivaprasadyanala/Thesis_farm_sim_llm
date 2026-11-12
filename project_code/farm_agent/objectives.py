"""Objective selection: what the episode is scored on.

The ids are never hard-coded here - they come from the running server
(`GET /api/v1/objectives`), so a new objective in `sim/systems/objective.js`
reaches this harness without an edit. What this module does enforce is that the
run's summary names the same goal the server is scoring.
"""

import logging

from . import runtime
from .api import api_get

logger = logging.getLogger(__name__)


def load_objective_catalogue():
    """Objectives the server accepts, and the one each preset assigns by default.

    ``POST /api/v1/reset {objective}`` takes any id from
    ``sim/systems/objective.js``, so the catalogue is read from the server instead
    of being duplicated here. A server started before ``GET /api/v1/objectives``
    existed still answers ``/api/v1/config``, whose presets name their default
    objective - which is enough to validate a choice.
    """
    runtime.objective_catalog = []
    runtime.preset_objectives = {}
    try:
        runtime.objective_catalog = api_get("/api/v1/objectives")["objectives"]
    except Exception as exc:
        print(f"note: this server does not list its objectives ({exc}); "
              f"falling back to the preset defaults of /api/v1/config")
    try:
        presets = api_get("/api/v1/config")["presets"]
        runtime.preset_objectives = {name: preset.get("objective")
                                     for name, preset in presets.items()}
    except Exception as exc:
        print(f"note: cannot read the preset catalogue either ({exc})")
    if not runtime.objective_catalog:
        ids = sorted({obj_id for obj_id in runtime.preset_objectives.values() if obj_id})
        runtime.objective_catalog = [{"id": obj_id, "text": "", "deadline_day": "?", "reward": "?"}
                                     for obj_id in ids]
    return runtime.objective_catalog


def print_objectives():
    """`--list-objectives`: what the server can score and which preset picks what."""
    load_objective_catalogue()
    if not runtime.objective_catalog:
        print("the server lists no objectives - is the farm-sim server reachable?")
        return
    print(f"{'id':<16} {'deadline':>8}  {'reward':>6}  objective")
    for entry in runtime.objective_catalog:
        print(f"{entry['id']:<16} {str(entry.get('deadline_day', '?')):>8}  "
              f"${str(entry.get('reward', '?')):>5}  {entry.get('text', '')}")
    print("")
    print("default objective per preset:")
    for name, obj_id in runtime.preset_objectives.items():
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
    if not runtime.objective_catalog:
        load_objective_catalogue()
    ids = [entry["id"] for entry in runtime.objective_catalog]
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
    default = runtime.preset_objectives.get(args.preset) or "?"
    if not args.objective:
        return f"objective: {default} (the {args.preset} preset default)"
    line = f"objective: {args.objective} (requested; {args.preset} preset default: {default})"
    if args.task:
        line += (f"\nnote: --task only rewrites the instruction text - "
                 f"{args.objective} is what the server scores")
    return line


def verify_episode_objective(args):
    """Check that the running episode scores the goal this run claims to score.

    Called from `farm_agent.episode.prepare_episode`, i.e. after `/api/v1/reset`
    or after adopting a running episode with `--no-reset`. It is the guard against
    the defect that made the recorded logs unusable: a run whose summary named one
    objective while the server was scoring another (a leftover episode, or an id
    the server did not know and replaced with `starter_easy`).
    """
    actual = (runtime.state.get("objective") or {}).get("id")
    if args.objective:
        expected = args.objective
    elif args.no_reset:
        expected = None          # the running episode decides; --objective is the only contract
    else:
        expected = runtime.preset_objectives.get(args.preset)
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
