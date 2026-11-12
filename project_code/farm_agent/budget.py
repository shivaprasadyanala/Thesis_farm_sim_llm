"""Game time: tick/minute conversion, the `budget:` line and the `advance` clamp.

Everything here reads the budget numbers out of `farm_agent.runtime` so the one
place that changes them (`--max-advance-ticks`, `--max-steps`,
`/api/v1/config`) is also the one place that defines them.
"""

import logging

from . import runtime

logger = logging.getLogger(__name__)


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
            return max(0, int(minutes) // runtime.MINUTES_PER_TICK)
        except (TypeError, ValueError):
            logger.info(f"advance: unusable minutes={minutes!r} - treated as 0 ticks")
            return 0
    return 0


def ticks_until_deadline():
    """Ticks left before the objective deadline day ends (None when unknown)."""
    objective = runtime.state.get("objective") or {}
    time_block = runtime.state.get("time") or {}
    deadline = objective.get("deadline_day")
    day, minute = time_block.get("day"), time_block.get("minute")
    if deadline is None or day is None or minute is None:
        return None
    remaining_minutes = ((int(deadline) - int(day) + 1) * runtime.MINUTES_PER_DAY - int(minute))
    return max(0, remaining_minutes // runtime.MINUTES_PER_TICK)


def budget_to_text():
    """One line of budget telemetry: steps left, game time left, advance cap.

    Both numbers are known to the harness and were invisible to the model. In the
    recorded `hard` run it spent five of its last ten steps recovering energy
    without knowing only five were left; in `normal` it chained nine `advance`
    calls (each clipped to one day) and hit the deadline with twelve steps unused.
    """
    parts = []
    if runtime.MAX_STEPS > 0:
        parts.append(f"{max(0, runtime.MAX_STEPS - runtime.STEPS_USED)} of "
                     f"{runtime.MAX_STEPS} step(s) left")
    remaining = ticks_until_deadline()
    if remaining is not None:
        days = remaining * runtime.MINUTES_PER_TICK / runtime.MINUTES_PER_DAY
        deadline = (runtime.state.get("objective") or {}).get("deadline_day")
        parts.append(f"{days:.1f} day(s) to the deadline (day {deadline})")
    if runtime.MAX_ADVANCE_TICKS > 0:
        per_call = runtime.MAX_ADVANCE_TICKS * runtime.MINUTES_PER_TICK / runtime.MINUTES_PER_DAY
        parts.append(f"one advance = at most {runtime.MAX_ADVANCE_TICKS} tick(s) = "
                     f"{per_call:.1f} day(s) and costs one step")
    return ("budget: " + " | ".join(parts)) if parts else ""


def clamp_advance(requested):
    """Keep one `advance` inside `--max-advance-ticks` and inside the deadline.

    Returns the ticks to actually send plus a note ('' when nothing was clipped)
    that is echoed to the model as `note`, so it can see the clock did not jump
    as far as it asked instead of assuming it did.
    """
    limit, source = None, ""
    if runtime.MAX_ADVANCE_TICKS > 0:
        limit, source = runtime.MAX_ADVANCE_TICKS, f"--max-advance-ticks {runtime.MAX_ADVANCE_TICKS}"
    to_deadline = ticks_until_deadline()
    if to_deadline is not None and (limit is None or to_deadline < limit):
        limit = to_deadline
        source = f"objective deadline (day {runtime.state['objective']['deadline_day']})"
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
