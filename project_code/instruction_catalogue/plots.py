"""Figures for a batch: task completion and the three latency clocks.

Reads the per-episode ``_summary.json`` files of a batch folder (never the logs -
the JSON is what the harness guarantees) and writes one set of PNGs per model:

* ``completion`` - how many episodes reached the objective, the mean score, and
  how the episodes ended (stop-reason mix);
* ``latency`` - the three clocks side by side: seconds per LLM request,
  milliseconds per tool call, seconds per episode;
* ``tradeoff`` - does a phrasing that scores better also cost more wall clock?

plus ``latency_completion.csv``, the aggregated numbers behind the figures (one
row per model x preset x task x level).

Two rules the figures follow, both inherited from `results`:

1. an episode the backend answered with ``llm_error`` never *played*, so it is
   excluded from the completion rate and from every latency mean - it is counted
   as ``err`` instead, which is why the bars are annotated ``k/n`` and not ``k/total``;
2. the three latency scopes are never mixed onto one axis: one chat completion,
   one ``POST /api/v1/action``, one whole agent loop are different units.

Two ways in, both ending in the same ``main``::

    python -m instruction_catalogue.plots      # from project_code
    python plots.py                            # from this folder, or from anywhere
"""

import argparse
import csv
import glob
import os
import re
import sys
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")  # experiment runs are headless
import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

if __package__ in (None, ""):
    # ``python plots.py``: a script has no parent package, so the sibling modules
    # are imported through the package - whose parent folder (``project_code``)
    # must be on sys.path regardless of the working directory.
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from instruction_catalogue import instructions, results
else:
    from . import instructions, results

#: ``<repo>/experiments/instruction_catalogue`` - where the documented batch
#: invocations write. Resolved from this file, not from the cwd, so a bare
#: ``python plots.py`` finds the episodes wherever it is started from.
DEFAULT_LOG_FOLDER = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "experiments", "instruction_catalogue")

#: ``<model>_<preset>_seed<seed>_<task>_<level>_p<prompt>_r<repeat>`` - the stem
#: ``Trial.artefact_stem`` gives every episode, so the summary alone (which records
#: the prompt text, not its index) can still be placed in the matrix.
EPISODE_PATTERN = re.compile(
    r"_(?P<preset>default|easy|normal|hard)_seed(?P<seed>\d+)_"
    r"(?P<task>[A-Za-z0-9_]+)_(?P<level>direct|indirect|implicit)"
    r"_p(?P<prompt>\d+)_r(?P<repeat>\d+)$")

#: Short axis labels; the preset travels in parentheses next to them.
TASK_SHORT = {
    "plant_crop_water_task": "plant + water",
    "harvest_crop_task": "harvest",
    "harvest_crop_earn_money_task": "harvest + earn",
}

#: One colour per directness level, catalogue order, used by every figure.
LEVEL_COLORS = OrderedDict([("direct", "#1f77b4"),
                            ("indirect", "#ff7f0e"),
                            ("implicit", "#2ca02c")])

#: Stop reasons, in stack order, and their colours. Everything that is not a
#: stopping rule of its own ends up in ``other``.
STOP_COLORS = OrderedDict([
    ("objective_complete", "#2ca02c"),
    ("objective_failed", "#d62728"),
    ("max_steps", "#ff7f0e"),
    ("llm_error", "#7f7f7f"),
    ("check_api_fail", "#c7c7c7"),
    ("other", "#1f77b4"),
])

PRESET_ORDER = ("default", "easy", "normal", "hard")


def parse_stem(stem):
    """The matrix coordinates encoded in an artefact stem, or ``None``."""
    match = EPISODE_PATTERN.search(stem)
    if match is None:
        return None
    found = match.groupdict()
    return {"preset": found["preset"], "seed": int(found["seed"]),
            "task": found["task"], "level": found["level"],
            "prompt_index": int(found["prompt"]), "repeat": int(found["repeat"])}


def _model_from_path(path, root):
    """The per-model folder name of ``path``, or ``unknown`` for a flat folder."""
    folder = os.path.dirname(os.path.abspath(path))
    if folder == os.path.abspath(root):
        return "unknown"
    return os.path.basename(folder) or "unknown"


def collect(log_folder):
    """Every ``*_summary.json`` under ``log_folder`` as one flat dict per episode.

    The folder may be one model's folder or the batch root (``**`` walks the
    per-model sub-folders); the batch aggregates ``_catalogue_*_summary.json`` are
    skipped, they are not episodes.
    """
    pattern = os.path.join(log_folder, "**", "*_summary.json")
    episodes = []
    for path in sorted(glob.glob(pattern, recursive=True)):
        name = os.path.basename(path)
        if name.startswith("_catalogue_"):
            continue
        meta = parse_stem(name[:-len("_summary.json")])
        if meta is None:
            print(f"note: {name} is not an episode summary - skipped")
            continue
        summary = results.load_summary(path)
        if summary is None:
            continue
        metrics = summary.get("metrics") or {}
        llm_stats = summary.get("llm_time_taken_stats") or {}
        tool_stats = summary.get("time_taken_stats") or {}
        stop_reason = summary.get("stop_reason")
        tool_calls = summary.get("no_of_tool_calls")
        if not tool_calls:
            tool_calls = len(summary.get("tool_calls") or [])
        episode = dict(meta)
        episode.update({
            # The child records its own model id; a `--check-api` scripted episode
            # leaves it null and is named after its per-model folder instead.
            "model": summary.get("model") or _model_from_path(path, log_folder),
            "summary": path,
            "points": summary.get("points_gained"),
            "completed": bool(metrics.get("objective_completed")),
            "stop_reason": stop_reason,
            "played": stop_reason not in results.FAILED_STOP_REASONS,
            "llm_calls": summary.get("no_of_llm_calls") or 0,
            "tool_calls": tool_calls,
            "s_llm": llm_stats.get("mean"),
            "s_tool": tool_stats.get("mean"),
            "s_episode": summary.get("episode_time"),
        })
        episodes.append(episode)
    return episodes


def _mean(values):
    numbers = [value for value in values if isinstance(value, (int, float))]
    return sum(numbers) / len(numbers) if numbers else None


def _sem(values):
    """Standard error of the mean (0 for a single value, ``None`` for none)."""
    numbers = [value for value in values if isinstance(value, (int, float))]
    if not numbers:
        return None
    if len(numbers) == 1:
        return 0.0
    mean = sum(numbers) / len(numbers)
    variance = sum((value - mean) ** 2 for value in numbers) / (len(numbers) - 1)
    return (variance / len(numbers)) ** 0.5


def _cell_stats(items):
    """The numbers one bar stands for: completion, score, the three clocks."""
    played = [item for item in items if item["played"]]
    completed = [item for item in played if item["completed"]]
    stop_counts = OrderedDict()
    for item in items:
        key = item["stop_reason"] if item["stop_reason"] in STOP_COLORS else "other"
        stop_counts[key] = stop_counts.get(key, 0) + 1

    llm_totals = [item["llm_calls"] * item["s_llm"] for item in played
                  if isinstance(item["s_llm"], (int, float))]
    tool_totals = [item["tool_calls"] * item["s_tool"] for item in played
                   if isinstance(item["s_tool"], (int, float))]
    return {
        "n": len(items),
        "played": len(played),
        "errors": len(items) - len(played),
        "completed": len(completed),
        "completion_rate": (100.0 * len(completed) / len(played) if played else None),
        "stop_counts": stop_counts,
        "mean_points": _mean([item["points"] for item in played]),
        "mean_llm_calls": _mean([item["llm_calls"] for item in played]),
        "mean_tool_calls": _mean([item["tool_calls"] for item in played]),
        "mean_s_llm": _mean([item["s_llm"] for item in played]),
        "mean_s_tool": _mean([item["s_tool"] for item in played]),
        "mean_s_episode": _mean([item["s_episode"] for item in played]),
        "sem_s_episode": _sem([item["s_episode"] for item in played]),
        "mean_llm_seconds": _mean(llm_totals),
        "mean_tool_seconds": _mean(tool_totals),
        "episodes": items,
    }


def aggregate(episodes):
    """``results[model][(preset, task)][level]`` -> `_cell_stats` of that cell.

    Ordered by preset, then by catalogue task order, then by level, so the figures
    of two models (or two batches) line up bar for bar.
    """
    task_order = list(instructions.TASK_OBJECTIVES)

    def task_rank(task):
        return task_order.index(task) if task in task_order else len(task_order)

    def preset_rank(preset):
        return (PRESET_ORDER.index(preset) if preset in PRESET_ORDER
                else len(PRESET_ORDER))

    grouped = {}
    for episode in episodes:
        cell = (episode["preset"], episode["task"])
        grouped.setdefault(episode["model"], {}).setdefault(cell, {})
        grouped[episode["model"]][cell].setdefault(episode["level"], []).append(episode)

    models = OrderedDict()
    for model, cells in grouped.items():
        ordered = OrderedDict()
        for cell in sorted(cells, key=lambda item: (preset_rank(item[0]),
                                                    task_rank(item[1]), item[1])):
            levels = OrderedDict()
            for level in level_order(cells[cell]):
                levels[level] = _cell_stats(cells[cell][level])
            ordered[cell] = levels
        models[model] = ordered
    return models


def _bar_offset(count, index, step):
    """x offset of bar ``index`` of ``count`` around the group's centre."""
    return (index - (count - 1) / 2.0) * step


def _clean(axis, ylabel):
    """Shared cosmetics: y grid, no top/right spine, a labelled y axis."""
    axis.set_ylabel(ylabel, fontsize=9)
    axis.grid(axis="y", linestyle=":", linewidth=0.6, color="0.75")
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def _label_bars(axis, bars, texts, anchor=None, fontsize=7):
    """Value above each bar; ``anchor`` lifts it over an error bar."""
    positions = anchor if anchor is not None else [bar.get_height() for bar in bars]
    for bar, text, position in zip(bars, texts, positions):
        if not text:
            continue
        axis.annotate(text, (bar.get_x() + bar.get_width() / 2.0, position),
                      xytext=(0, 2), textcoords="offset points",
                      ha="center", va="bottom", fontsize=fontsize)


def _headroom(axis, factor=1.18):
    """Leave room for the value labels above the tallest bar."""
    bottom, top = axis.get_ylim()
    axis.set_ylim(bottom, top * factor if top > 0 else 1.0)


def _step(level_count):
    """Bar width: the group is 0.8 wide, but one level alone stays bar-sized.

    A folder with a single wording (a `--check-api` smoke round, or a model that
    only ran one level) would otherwise draw one bar 0.8 wide across a 1.2 wide
    axis - a slab, not a bar.
    """
    return min(0.8 / max(1, level_count), 0.28)


def _plain_ticks(axis, values):
    """Plain x tick labels on a log axis: ``310``, not ``3.1 x 10^2``.

    Two regimes, because matplotlib decides differently: under one decade it
    labels the sub-decade ticks (as *minor* ticks, in scientific notation), over
    several decades it labels only the powers of ten. The first gets plain numbers
    on the minors, the second a 1-2-5 major grid (``5, 10, 20, 50, 100``) - the
    scale itself stays logarithmic either way, the axis label says so.
    """
    formatter = mticker.FuncFormatter(
        lambda value, _: f"{value:,.0f}" if value >= 10 else f"{value:g}")
    axis.xaxis.set_major_formatter(formatter)
    positive = sorted(value for value in values if value and value > 0)
    if not positive:
        return
    if positive[-1] < 10 * positive[0]:
        axis.xaxis.set_minor_formatter(formatter)
    else:
        axis.xaxis.set_major_locator(
            mticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0), numticks=12))


def _save(figure, path, dpi):
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def plot_completion(model, cells, path, dpi=150):
    """Completion rate, mean score and the stop-reason mix, per cell and level.

    Three panels, same x axis: the bar height answers "did it reach the goal", the
    annotation ``k/n`` counts the episodes that *played* (an ``llm_error`` episode
    is reported as ``k err`` instead of silently counting as a failure), and the
    stack says *why* the rest stopped.
    """
    groups = list(cells)
    levels = levels_of(cells)
    step = _step(len(levels))
    positions = list(range(len(groups)))
    offsets = {level: [position + _bar_offset(len(levels), index, step)
                       for position in positions]
               for index, level in enumerate(levels)}

    figure, axes = plt.subplots(1, 3, figsize=(17.5, 5.0), constrained_layout=True)
    figure.suptitle(f"{model} - task completion by instruction directness", fontsize=12)

    for level in levels:
        color = LEVEL_COLORS.get(level, "#333333")
        stats = [cells[group].get(level) for group in groups]
        rates = [cell["completion_rate"] if cell and cell["completion_rate"] is not None
                 else 0.0 for cell in stats]
        texts = []
        for index, cell in enumerate(stats):
            if not cell:
                texts.append(None)
                continue
            label = f"{cell['completed']}/{cell['played']}"
            texts.append(label + (f"\nerr {cell['errors']}" if cell["errors"] else ""))
        bars = axes[0].bar(offsets[level], rates, step, label=level, color=color,
                           edgecolor="white")
        _label_bars(axes[0], bars, texts,
                    anchor=[max(rate, 5.0) for rate in rates], fontsize=6)

        points = [cell["mean_points"] if cell and cell["mean_points"] is not None else 0.0
                  for cell in stats]
        bars = axes[1].bar(offsets[level], points, step, label=level, color=color,
                           edgecolor="white")
        _label_bars(axes[1], bars,
                    [f"{value:.0f}" if cell and cell["mean_points"] is not None else None
                     for value, cell in zip(points, stats)])

        bottom = [0.0] * len(groups)
        for reason, reason_color in STOP_COLORS.items():
            shares = [100.0 * cell["stop_counts"].get(reason, 0) / cell["n"] if cell else 0.0
                      for cell in stats]
            if not any(shares):
                continue
            axes[2].bar(offsets[level], shares, step, bottom=bottom, color=reason_color,
                        edgecolor="white", linewidth=0.4)
            bottom = [low + share for low, share in zip(bottom, shares)]

    for axis in axes:
        axis.set_xticks(positions)
        axis.set_xticklabels([cell_label(group) for group in groups], fontsize=8,
                             rotation=12, ha="right", rotation_mode="anchor")
        axis.set_xlim(-0.6, len(groups) - 0.4)
    _clean(axes[0], "% episodes completed")
    _clean(axes[1], "mean score (points)")
    _clean(axes[2], "% of episodes")
    axes[0].set_title("goal reached (completed / played, err = llm_error)",
                      fontsize=10)
    axes[1].set_title("mean score", fontsize=10)
    axes[2].set_title("how the episodes ended", fontsize=10)
    axes[0].set_ylim(0, 118)
    scores = [cell["mean_points"] for levels in cells.values()
              for cell in levels.values()
              if cell and cell["mean_points"] is not None]
    axes[1].set_ylim(0, max(scores + [100.0]) * 1.15)
    axes[2].set_ylim(0, 100)
    axes[0].legend(title="wording", fontsize=8, title_fontsize=8, frameon=False,
                   loc="upper center", ncol=len(levels), bbox_to_anchor=(0.5, -0.12))
    handles = [Patch(color=color, label=reason.replace("_", " "))
               for reason, color in STOP_COLORS.items()
               if any(cell and cell["stop_counts"].get(reason)
                      for levels in cells.values() for cell in levels.values())]
    axes[2].legend(handles=handles, fontsize=7, frameon=False, loc="upper center",
                   ncol=2, bbox_to_anchor=(0.5, -0.12))
    return _save(figure, path, dpi)


def plot_latency(model, cells, path, dpi=150):
    """The three clocks, one panel each - never on a shared axis.

    A bar is the mean of the cell's repeats; the episode panel carries a SEM
    whisker because that is the scope whose spread a reader needs, and every bar is
    annotated with its own value. ``mean_s_tool`` is shown in milliseconds: a round
    trip to the farm server is ~20 ms here, so seconds would flatten it to 0.0.
    """
    groups = list(cells)
    levels = levels_of(cells)
    step = _step(len(levels))
    positions = list(range(len(groups)))
    offsets = {level: [position + _bar_offset(len(levels), index, step)
                       for position in positions]
               for index, level in enumerate(levels)}
    specs = (("mean_s_llm", 1.0, "{:.2f}", "seconds",
              "one LLM request (client-side clock)"),
             ("mean_s_tool", 1000.0, "{:.1f}", "milliseconds",
              "one tool call (server round trip)"),
             ("mean_s_episode", 1.0, "{:.1f}", "seconds",
              "one episode (whole agent loop)"))

    figure, axes = plt.subplots(3, 1, figsize=(11.5, 10.8), sharex=True,
                                constrained_layout=True)
    figure.suptitle(f"{model} - latency per scope (mean over the repeats)",
                    fontsize=12)

    for axis, (key, scale, fmt, unit, title) in zip(axes, specs):
        recorded = False
        for level in levels:
            color = LEVEL_COLORS.get(level, "#333333")
            stats = [cells[group].get(level) for group in groups]
            heights = [cell[key] * scale if cell and cell[key] is not None else 0.0
                       for cell in stats]
            recorded = recorded or any(heights)
            errors = None
            if key == "mean_s_episode":
                errors = [(cell["sem_s_episode"] or 0.0) * scale if cell else 0.0
                          for cell in stats]
            bars = axis.bar(offsets[level], heights, step, label=level, color=color,
                            edgecolor="white", yerr=errors, capsize=2,
                            error_kw={"elinewidth": 0.8, "ecolor": "0.3"})
            anchors = ([height + (error or 0.0) for height, error in zip(heights, errors)]
                       if errors else None)
            _label_bars(axis, bars, [fmt.format(value) if value else None
                                     for value in heights], anchor=anchors)
        _clean(axis, unit)
        axis.set_title(title, fontsize=10)
        _headroom(axis)
        if not recorded:
            # An empty panel must say so: a scripted `--check_api` round has no LLM
            # clock, and a backend that only answered 403 has no timings either.
            axis.text(0.5, 0.5, "no timings recorded", transform=axis.transAxes,
                      ha="center", va="center", fontsize=10, color="0.45")
            axis.set_ylim(0, 1)
    axes[0].legend(title="wording", fontsize=8, title_fontsize=8, frameon=False,
                   loc="upper left", ncol=len(levels))
    axes[-1].set_xticks(positions)
    axes[-1].set_xticklabels([cell_label(group) for group in groups], fontsize=8,
                             rotation=12, ha="right", rotation_mode="anchor")
    axes[-1].set_xlim(-0.6, len(groups) - 0.4)
    return _save(figure, path, dpi)


def plot_tradeoff(model, cells, path, dpi=150):
    """Score against wall clock, per episode (left) and per cell (right).

    Left: every episode that played, episode seconds (log) against score; a filled
    marker reached the objective, a hollow one did not. Right: one point per cell
    and level - mean episode seconds against completion rate, the levels of a cell
    joined by a dashed line, so the direction the wording moves the trade-off in is
    visible at a glance.

    ``llm_error`` episodes are in neither panel: they never reached the model, so
    their 0 points and ~0 seconds would read as a free failure.
    """
    levels = levels_of(cells)
    figure, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), constrained_layout=True)
    figure.suptitle(f"{model} - does the better-scoring wording also cost more "
                    f"wall clock?", fontsize=12)

    episode_times = []
    for level in levels:
        color = LEVEL_COLORS.get(level, "#333333")
        played = [item for level_cells in cells.values() if level in level_cells
                  for item in level_cells[level]["episodes"]
                  if item["played"] and isinstance(item["s_episode"], (int, float))
                  and item["s_episode"] > 0]
        episode_times.extend(item["s_episode"] for item in played)
        done = [item for item in played if item["completed"]]
        missed = [item for item in played if not item["completed"]]
        if done:
            axes[0].scatter([item["s_episode"] for item in done],
                            [item["points"] or 0 for item in done],
                            s=26, color=color, alpha=0.85, linewidth=0)
        if missed:
            axes[0].scatter([item["s_episode"] for item in missed],
                            [item["points"] or 0 for item in missed],
                            s=30, facecolor="none", edgecolor=color, linewidth=0.9)
    axes[0].set_xscale("log")
    _plain_ticks(axes[0], episode_times)
    axes[0].set_xlabel("episode time (s, log scale)", fontsize=9)
    axes[0].set_title("every episode (filled = objective reached)", fontsize=10)
    _clean(axes[0], "score (points)")
    _headroom(axes[0])
    # Proxies, not the scatters: a folder whose episodes all failed would leave a
    # labelled-scatter legend empty (and matplotlib warns about it).
    handles = [Line2D([], [], marker="o", linestyle="", markersize=6,
                      color=LEVEL_COLORS.get(level, "#333333"), label=level)
               for level in levels]
    axes[0].legend(handles=handles, title="wording", fontsize=8, title_fontsize=8,
                   frameon=False, loc="upper left")

    anchors = []
    cell_times = []
    for cell, level_cells in cells.items():
        points = []
        for level in levels:
            cell_stats = level_cells.get(level)
            if not cell_stats or cell_stats["mean_s_episode"] is None:
                continue
            if cell_stats["completion_rate"] is None:
                continue
            points.append((level, cell_stats))
        if not points:
            continue
        xs = [cell_stats["mean_s_episode"] for _, cell_stats in points]
        ys = [cell_stats["completion_rate"] for _, cell_stats in points]
        cell_times.extend(xs)
        axes[1].plot(xs, ys, color="0.6", linestyle="--", linewidth=0.8, zorder=1)
        for level, cell_stats in points:
            axes[1].scatter(cell_stats["mean_s_episode"], cell_stats["completion_rate"],
                            s=20 + 1.2 * (cell_stats["mean_llm_calls"] or 0),
                            color=LEVEL_COLORS.get(level, "#333333"),
                            edgecolor="white", linewidth=0.6, zorder=2)
        anchors.append({"codes": [cell_code(cell)], "x": xs[ys.index(max(ys))],
                        "y": max(ys)})

    # Two cells that end at the same rate and seconds apart would label over each
    # other (``plant + water`` at easy and at normal both sit at 100 %), so the
    # labels of near-identical cells are merged into one, with a leader line to the
    # cluster they belong to.
    merged = []
    for anchor in anchors:
        previous = merged[-1] if merged else None
        if (previous is not None
                and abs(anchor["x"] - previous["x"]) <= 0.08 * previous["x"]
                and abs(anchor["y"] - previous["y"]) <= 5):
            previous["codes"].extend(anchor["codes"])
            previous["x"] = (previous["x"] + anchor["x"]) / 2.0
            continue
        merged.append(anchor)
    for anchor in merged:
        # Label away from the middle of the panel - above a high point, below a
        # mid-low one - and above a ~0 % point, where below is under the axis.
        above = anchor["y"] >= 50 or anchor["y"] < 8
        axes[1].annotate(", ".join(anchor["codes"]), (anchor["x"], anchor["y"]),
                         xytext=(0, 14 if above else -16), textcoords="offset points",
                         ha="center", va="bottom" if above else "top", fontsize=7,
                         color="0.35",
                         arrowprops={"arrowstyle": "-", "color": "0.6", "lw": 0.6,
                                     "shrinkA": 0, "shrinkB": 3})
    axes[1].set_xscale("log")
    _plain_ticks(axes[1], cell_times)
    axes[1].set_xlabel("mean episode time (s, log scale)", fontsize=9)
    axes[1].set_title("one point per cell and wording "
                      "(marker size = mean LLM calls)", fontsize=10)
    _clean(axes[1], "completion rate (%)")
    axes[1].set_ylim(-18, 126)
    return _save(figure, path, dpi)


#: Columns of ``latency_completion.csv`` - the aggregated numbers behind the bars.
AGGREGATE_FIELDS = ("model", "preset", "task", "level", "episodes", "played", "errors",
                    "completed", "completion_rate", "mean_points", "mean_llm_calls",
                    "mean_tool_calls", "mean_s_llm", "mean_s_tool", "mean_s_episode",
                    "sem_s_episode", "mean_llm_seconds", "mean_tool_seconds")


def write_table(path, model, cells):
    """One row per model x preset x task x level, rounded floats, ``None`` holes."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(AGGREGATE_FIELDS))
        writer.writeheader()
        for (preset, task), levels in cells.items():
            for level, cell_stats in levels.items():
                row = {"model": model, "preset": preset, "task": task,
                       "level": level, "episodes": cell_stats["n"],
                       "played": cell_stats["played"], "errors": cell_stats["errors"],
                       "completed": cell_stats["completed"]}
                for field in AGGREGATE_FIELDS:
                    if field in row:
                        continue
                    value = cell_stats.get(field)
                    row[field] = round(value, 3) if isinstance(value, float) else value
                writer.writerow(row)
    return path


def write(log_folder, out_folder=None, dpi=150):
    """Every figure and the table of a batch folder; returns the written paths.

    The figures are per model (a batch may hold several), written into
    ``<log_folder>/figures`` unless ``out_folder`` says otherwise, and named after
    the sanitised model id so two models never overwrite each other.
    """
    log_folder = os.path.abspath(log_folder)
    episodes = collect(log_folder)
    if not episodes:
        print(f"no episode summaries under {log_folder} - nothing to plot")
        return []
    models = aggregate(episodes)
    out_folder = os.path.abspath(out_folder or os.path.join(log_folder, "figures"))
    os.makedirs(out_folder, exist_ok=True)

    paths = []
    for model, cells in models.items():
        stem = os.path.join(out_folder, instructions.sanitise(model))
        paths.append(plot_completion(model, cells, f"{stem}_completion.png", dpi=dpi))
        paths.append(plot_latency(model, cells, f"{stem}_latency.png", dpi=dpi))
        paths.append(plot_tradeoff(model, cells, f"{stem}_tradeoff.png", dpi=dpi))
        paths.append(write_table(f"{stem}_latency_completion.csv", model, cells))
        bars = sum(len(levels) for levels in cells.values())
        plotted = sum(cell["n"] for levels in cells.values() for cell in levels.values())
        print(f"{model}: {plotted} episode(s), {len(cells)} preset/task "
              f"pair(s), {bars} bar(s)")
    print(f"figures: {out_folder}")
    for path in paths:
        print(f"  {os.path.basename(path)}")
    return paths


def main(argv=None):
    """``python plots.py [--log-folder <folder>]`` - the same from the module.

    Re-plots a finished batch: no server, no LLM, no catalogue file - only the
    ``_summary.json`` files that are already there. Without ``--log-folder`` the
    batch folder of this repository is used, so a bare ``python plots.py``
    redraws the figures of ``experiments/instruction_catalogue``.
    """
    parser = argparse.ArgumentParser(
        prog="plots.py",
        description="Plot task completion and the three latency clocks of a "
                    "catalogue batch folder.")
    parser.add_argument("--log-folder", default=DEFAULT_LOG_FOLDER,
                        help="batch folder (or one model's sub-folder) to read "
                             "(default: <repo>/experiments/instruction_catalogue)")
    parser.add_argument("--out-folder", default=None,
                        help="where the PNGs/CSV go (default: <log-folder>/figures)")
    parser.add_argument("--dpi", type=int, default=150, help="PNG resolution")
    args = parser.parse_args(argv)
    return 0 if write(args.log_folder, args.out_folder, args.dpi) else 1


def level_order(present):
    """``direct, indirect, implicit`` first (catalogue order), then any extras."""
    known = [level for level in instructions.LEVELS if level in present]
    return known + sorted(set(present) - set(instructions.LEVELS))


def levels_of(cells):
    """The levels present in a ``{cell: {level: ...}}`` mapping, in `level_order`."""
    return level_order({level for level_cells in cells.values()
                        for level in level_cells})


def cell_label(cell):
    """``plant + water (easy)`` - the axis tick of one preset/task pair."""
    preset, task = cell
    return f"{TASK_SHORT.get(task, task)} ({preset})"


#: One letter per task group, for the annotation of the trade-off panel.
TASK_CODE = {"plant_crop_water_task": "P", "harvest_crop_task": "H",
             "harvest_crop_earn_money_task": "M"}


def cell_code(cell):
    """``P-E`` - task initial and preset initial (``plant + water`` at ``easy``)."""
    preset, task = cell
    return f"{TASK_CODE.get(task, '?')}-{preset[0].upper()}"


if __name__ == "__main__":
    raise SystemExit(main())
