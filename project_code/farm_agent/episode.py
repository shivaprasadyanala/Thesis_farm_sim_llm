"""Episode lifecycle: start (or adopt) the world, then verify what it scores."""

import logging

from . import runtime
from .api import api_post, refresh_state
from .objectives import verify_episode_objective
from .tracking import reset_tracking

logger = logging.getLogger(__name__)


def prepare_episode(args):
    """Start a fresh episode (or adopt the running one) and track its start tile."""
    reset_tracking()
    if args.no_reset:
        refresh_state()
        print(f"keeping running episode: preset={runtime.state['preset']} "
              f"seed={runtime.state['seed']} tick={runtime.state['tick']} "
              f"objective={runtime.state['objective']['id']}")
        logger.info(f"episode kept: preset={runtime.state['preset']} "
                    f"seed={runtime.state['seed']} tick={runtime.state['tick']} "
                    f"objective={runtime.state['objective']['id']}")
        # A paused world rejects every action with `PAUSED`, which looks exactly
        # like an agent that does nothing - resume it before the run.
        if runtime.state["time"].get("paused"):
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
        runtime.state = result["state"]
        print(f"episode {result['episode']} preset={result['preset']} seed={result['seed']} "
              f"objective={runtime.state['objective']['id']}")
        logger.info(f"episode reset: episode={result['episode']} preset={result['preset']} "
                    f"seed={result['seed']} objective={runtime.state['objective']['id']}")
    runtime.previous_tile = (int(runtime.state["player"]["tile"][0]),
                             int(runtime.state["player"]["tile"][1]))
    runtime.visited.add(runtime.previous_tile)
    runtime.player_positions.append(list(runtime.previous_tile))
    # Never let a stale goal survive into the run: the summary and the log have to
    # describe the objective the server is actually scoring (`--no-reset` and an
    # unknown id are the two ways a previous run's goal used to leak in).
    exit_code = verify_episode_objective(args)
    if exit_code:
        raise SystemExit(exit_code)
    return runtime.state


def resolve_task(args):
    """Task text the prompt asks for: `--task`, else the episode's objective.

    `state["task"]` already mirrors the objective unless `--task` overrode it, so
    the system prompt, the first user message and the HUD all agree with the
    objective the server scores. Without this the prompt's task block was empty
    and the model only saw the objective buried in the state dump.
    """
    if args.task:
        return args.task
    objective = runtime.state.get("objective") or {}
    text = objective.get("text") or runtime.state.get("task") or "farm the field"
    deadline = objective.get("deadline_day", "?")
    return f"{text} The episode ends when the objective is complete (it fails after day {deadline})."
