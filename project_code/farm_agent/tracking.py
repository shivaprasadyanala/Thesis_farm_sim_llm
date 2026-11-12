"""Per-episode bookkeeping: visited tiles, jumps, revisits and the reset.

The counters themselves live in `farm_agent.runtime` (they are read by the
summary, the `budget:` line and the tools); this module only knows how to move
them.
"""

import logging

from . import runtime

logger = logging.getLogger(__name__)


def reset_tracking():
    """Zero every counter the summary reports; called once per episode."""
    runtime.state = {}
    runtime.visited = set()
    runtime.player_positions = []
    runtime.previous_tile = None
    runtime.invalid_moves = 0
    runtime.invalid_move_object = {}
    runtime.revisits = 0
    runtime.jumps = 0
    runtime.tool_calls = []
    runtime.time_taken = []
    runtime.total_input_tokens = 0
    runtime.total_output_tokens = 0
    runtime.stop_reason = ""
    runtime.STEPS_USED = 0


def track_position(action):
    """Record the farmer tile after an action: revisits and unexplained jumps."""
    if not runtime.state:
        return
    tile = (int(runtime.state["player"]["tile"][0]), int(runtime.state["player"]["tile"][1]))
    if runtime.previous_tile is not None and tile != runtime.previous_tile:
        step = abs(tile[0] - runtime.previous_tile[0]) + abs(tile[1] - runtime.previous_tile[1])
        if step > 1 and action != "move_to":
            # `move_to` walks a whole A* path, so only other actions count.
            runtime.jumps += 1
            print(f"Jump of {step} tiles: {runtime.previous_tile} -> {tile} ({action})")
        if tile in runtime.visited:
            runtime.revisits += 1
    runtime.visited.add(tile)
    runtime.previous_tile = tile
    runtime.player_positions.append([tile[0], tile[1]])
