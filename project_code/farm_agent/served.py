"""Served constants - what the prompt is allowed to claim.

Everything the model is told about payoffs, litres, seasons and tick length is
read from ``GET /api/v1/config`` here, so a rebalanced sim reaches the prompt
without a code edit.
"""

import logging
import math
import re

import requests

from . import runtime
from .api import api_get
from .budget import ticks_until_deadline

logger = logging.getLogger(__name__)

#: Season codes for the crop table: four letters beat "spring,summer,autumn" in a
#: table the model re-reads on every step.
_SEASON_CODES = {"spring": "spr", "summer": "sum", "autumn": "aut", "winter": "win"}


def load_server_config():
    """Cache ``GET /api/v1/config`` for the prompt; never fatal.

    Everything the model is told about payoffs, litres and tick length comes from
    here, so a rebalanced sim reaches the prompt without a code edit. This is what
    the recorded `normal` run lacked: it planted four wheats - a crop whose four
    harvests gross about $115 - against a $150 objective, and its trace says
    "The harvest amount is not specified in the prompt" even though the server
    publishes `yield_kg` and `sell_price` for every crop.
    """
    try:
        runtime.server_config = api_get("/api/v1/config") or {}
    except requests.RequestException as exc:
        logger.warning(f"/api/v1/config unavailable ({exc}); prompt falls back to literal figures")
        runtime.server_config = {}
        return
    tick_minutes = runtime.server_config.get("minutes_per_tick")
    ticks_per_day = runtime.server_config.get("ticks_per_day")
    if isinstance(tick_minutes, (int, float)) and tick_minutes > 0:
        runtime.MINUTES_PER_TICK = tick_minutes
    if isinstance(ticks_per_day, (int, float)) and ticks_per_day > 0 and tick_minutes:
        # Derived, not assumed: "one advance = one day" is only true if the server
        # still runs 120 ticks of 12 minutes.
        runtime.MINUTES_PER_DAY = tick_minutes * ticks_per_day


def preset_value(name):
    """One served preset field for the episode's preset (None when absent)."""
    preset_name = (runtime.state or {}).get("preset")
    if not preset_name:
        preset_name = (runtime.server_config.get("current") or {}).get("preset")
    preset = (runtime.server_config.get("presets") or {}).get(preset_name) or {}
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
    value = runtime.server_config.get("days_per_season")
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


def crop_seasons_text(crop):
    """`spr,sum,aut` - the seasons a crop grows in (served by `catalogView`)."""
    seasons = crop.get("seasons") or []
    codes = [_SEASON_CODES.get(season, season[:3]) for season in seasons]
    return ",".join(codes) if codes else "?"


def crop_economics_to_text():
    """One line per crop: days, seed cost, price per kg, yield, gross $ and seasons.

    Without the yield the objective is unplannable: `$150 from 4 harvests` is
    impossible for wheat (~$29 each) and trivial for corn (~$75 each), and only
    the server knows which is which. The seasons column is served by
    `catalogView` and is what makes the `hard` preset's winter stall visible
    before the crop stops growing instead of after.
    """
    crops = runtime.server_config.get("crops") or {}
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


def objective_targets():
    """(earnings, harvests) the served objective text asks for; (None, None) if unknown.

    `/api/v1/config` publishes the objective's prose (`Earn $150 from sales and
    harvest 4 crops before day 18`) but not its numbers, so the two figures the
    plan needs are read out of that prose. Anything unparsable yields None and the
    PLAN block is shortened instead of guessing a target.
    """
    text = ((runtime.state or {}).get("objective") or {}).get("text") or ""
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
    progress = ((runtime.state or {}).get("objective") or {}).get("progress") or {}
    if "earnings" not in progress and "harvests" not in progress:
        return ""
    target_earnings, target_harvests = objective_targets()
    earned = progress.get("earnings") or 0
    harvests = progress.get("harvests") or 0
    crops = runtime.server_config.get("crops") or {}
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
        days_left = remaining * runtime.MINUTES_PER_TICK / runtime.MINUTES_PER_DAY
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
    days = [crop.get("days") for crop in (runtime.server_config.get("crops") or {}).values()
            if isinstance(crop.get("days"), (int, float))]
    if not days:
        return "3-5"
    lo, hi = min(days), max(days)
    return f"{lo:g}-{hi:g}" if lo != hi else f"{hi:g}"
