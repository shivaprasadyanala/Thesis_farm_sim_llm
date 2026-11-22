"""Collect the per-episode summaries into one table.

The episode script already writes everything an episode can be judged by into
``<stem>_summary.json`` - score, stop reason, tool sequence, latency statistics,
tokens. This module reads those files back (never the logs: the JSON is what the
harness guarantees) and groups them the way the pasted legacy script grouped its
runs, ``results[task][level]``, with the model added as the outer key:

    results[model][task][level] = {points: [...], completed: [...], ...}

Lists, not means: every phrasing is one data point, and a mean alone hides a
0/100/0 split. The derived means sit next to them for the console tables, and the
long-format CSV (one row per episode) is what gets read into pandas.

A missing episode is not an error here. ``--resume`` and ``--limit`` leave holes
in the matrix on purpose, so a hole is recorded as ``missing`` and the aggregate
still describes the episodes that do exist.
"""

import csv
import datetime
import json
import os
from collections import OrderedDict

from . import instructions

#: The per-cell lists, in the order they are written to the JSON.
CELL_LISTS = (
    "points",
    "completed",
    "tool_calls",
    "tool_names",
    "stop_reasons",
    "llm_calls",
    "time_taken_mean",
    "llm_time_taken_mean",
    "episode_time",
    "total_input_tokens",
    "total_output_tokens",
    "days_played",
    "summaries",
)


def load_summary(path):
    """The episode's ``_summary.json``, or ``None`` when it is absent/unreadable."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    except (ValueError, OSError) as exc:
        print(f"note: cannot read {path} ({exc})")
        return None


def _empty_cell():
    cell = OrderedDict((name, []) for name in CELL_LISTS)
    cell["n"] = 0
    cell["missing"] = 0
    return cell


def _mean(values):
    """Mean over the recorded numbers only (``None`` holes do not count)."""
    numbers = [value for value in values if isinstance(value, (int, float))]
    return round(sum(numbers) / len(numbers), 2) if numbers else None


def _finalise(results):
    """Add the derived per-cell statistics the console tables and CSV use."""
    for model_cells in results.values():
        for task_cells in model_cells.values():
            for cell in task_cells.values():
                recorded = sum(1 for point in cell["points"] if point is not None)
                cell["recorded"] = recorded
                cell["completed_count"] = sum(1 for done in cell["completed"] if done)
                cell["completed_rate"] = (round(cell["completed_count"] / recorded, 3)
                                          if recorded else None)
                cell["mean_points"] = _mean(cell["points"])
                cell["mean_tool_calls"] = _mean(cell["tool_calls"])
                cell["mean_llm_calls"] = _mean(cell["llm_calls"])
                cell["mean_time_taken"] = _mean(cell["time_taken_mean"])
                cell["mean_llm_time_taken"] = _mean(cell["llm_time_taken_mean"])
                cell["mean_episode_time"] = _mean(cell["episode_time"])
                cell["mean_days_played"] = _mean(cell["days_played"])


#: Columns of the long-format CSV, one row per episode. The three latency columns
#: are deliberately separate: `time_taken_mean` is one server action,
#: `llm_time_taken_mean` is one chat completion, `episode_time` is the whole loop.
ROW_FIELDS = ("model", "backend", "task", "level", "prompt_index", "repeat",
              "seed", "preset", "objective", "points", "completed", "stop_reason",
              "llm_calls", "tool_calls", "time_taken_mean", "llm_time_taken_mean",
              "episode_time", "days_played", "summary", "prompt")

#: Stop reasons that mean "the episode never got to play", as opposed to a
#: stopping rule (step budget, deadline, objective done). The batch reports these
#: as failures even though the child process exited 0: a model without tool
#: support answers every prompt with `llm_error`, and 300 such episodes must not
#: look like a finished experiment.
FAILED_STOP_REASONS = ("llm_error", "check_api_fail")


def _summary_label(path, log_folder):
    """The artefact name as recorded: relative to the batch folder when nested.

    A per-model sub-folder should stay visible in the CSV and the JSON (two models
    can produce the same stem only if their ids sanitise alike, which the batch
    refuses), so ``Qwen_Qwen3_8-27B/<stem>_summary.json`` is preferred over the
    bare file name. Paths outside ``log_folder`` fall back to the file name.
    """
    try:
        relative = os.path.relpath(path, log_folder)
    except ValueError:  # different drive on Windows
        return os.path.basename(path)
    if relative.startswith(os.pardir):
        return os.path.basename(path)
    return relative.replace(os.sep, "/")


def collect(trials, log_folder, group_models=True):
    """Group the summaries of ``trials`` into ``(results, rows)``.

    ``results[model][task][level]`` keeps the per-episode values as lists (so a
    0/100/0 split stays visible); ``rows`` is the same data flattened, one dict
    per episode, in `ROW_FIELDS` order. ``group_models`` mirrors the batch layout:
    with one sub-folder per model the ``summary`` column is the path relative to
    ``log_folder`` (``<model>/<stem>_summary.json``), so the CSV stays readable
    from the batch folder.
    """
    results = OrderedDict()
    rows = []
    for trial in trials:
        path = trial.summary_path(log_folder, group_models)
        summary = load_summary(path)
        cell = (results.setdefault(trial.model, OrderedDict())
                .setdefault(trial.task, OrderedDict())
                .setdefault(trial.level, _empty_cell()))
        cell["n"] += 1

        row = OrderedDict((field, None) for field in ROW_FIELDS)
        row.update([("model", trial.model), ("backend", trial.backend),
                    ("task", trial.task), ("level", trial.level),
                    ("prompt_index", trial.prompt_index), ("repeat", trial.repeat),
                    ("seed", trial.seed), ("preset", trial.preset),
                    ("objective", trial.objective), ("prompt", trial.prompt),
                    ("summary", _summary_label(path, log_folder))])

        if summary is None:
            cell["missing"] += 1
            for name in CELL_LISTS:
                cell[name].append(None)
        else:
            metrics = summary.get("metrics") or {}
            tool_calls = summary.get("no_of_tool_calls")
            if not tool_calls:
                # `--check-api` episodes never append to `time_taken`, which is what
                # `no_of_tool_calls` counts; the action list is the fallback that keeps
                # a smoke row from looking like an episode that did nothing.
                tool_calls = len(summary.get("tool_calls") or [])
            values = {
                "points": summary.get("points_gained"),
                "completed": bool(metrics.get("objective_completed")),
                "tool_calls": tool_calls,
                "tool_names": (summary.get("tool_calls_deduped")
                               or summary.get("tool_calls") or []),
                "stop_reasons": summary.get("stop_reason"),
                "llm_calls": summary.get("no_of_llm_calls"),
                "time_taken_mean": (summary.get("time_taken_stats") or {}).get("mean"),
                "llm_time_taken_mean": (summary.get("llm_time_taken_stats") or {}).get("mean"),
                "episode_time": summary.get("episode_time"),
                "total_input_tokens": summary.get("total_input_tokens"),
                "total_output_tokens": summary.get("total_output_tokens"),
                "days_played": summary.get("days_played"),
                "summaries": _summary_label(path, log_folder),
            }
            for name in CELL_LISTS:
                cell[name].append(values[name])
            row.update([("points", values["points"]),
                        ("completed", values["completed"]),
                        ("stop_reason", values["stop_reasons"]),
                        ("llm_calls", values["llm_calls"]),
                        ("tool_calls", values["tool_calls"]),
                        ("time_taken_mean", values["time_taken_mean"]),
                        ("llm_time_taken_mean", values["llm_time_taken_mean"]),
                        ("episode_time", values["episode_time"]),
                        ("days_played", values["days_played"])])
        rows.append(row)

    _finalise(results)
    return results, rows


def _cell_score(cell):
    """``"40.0 (3/3)"`` - mean points plus how many of the cell's episodes scored."""
    mean = cell.get("mean_points")
    done = cell.get("completed_count") or 0
    total = cell.get("recorded") or 0
    if mean is None:
        return f"- (0/{total})"
    return f"{mean:.1f} ({done}/{total})"


def _cell_number(cell, key):
    value = cell.get(key)
    return "-" if value is None else f"{value:.2f}"


def _print_table(task_cells, title, render):
    """One task x level table (rows: task groups, columns: directness levels)."""
    levels = list(OrderedDict.fromkeys(
        level for cells in task_cells.values() for level in cells))
    if not levels:
        return
    width = max([len(task) for task in task_cells] + [4])
    print(f"  {title}")
    print(f"    {'task':<{width}}  " + "  ".join(f"{level:>17}" for level in levels))
    for task, cells in task_cells.items():
        printed = [f"{render(cells[level]):>17}" if level in cells else f"{'-':>17}"
                   for level in levels]
        print(f"    {task:<{width}}  " + "  ".join(printed))


def print_tables(results):
    """Five tables per model: score, tool calls, LLM calls, seconds, seconds/episode.

    A phrasing that scores the same with fewer tool calls is a different result,
    so cost is printed next to the score instead of being left in the JSON. The
    three latency tables are separate scopes - one chat completion, one server
    action, one whole agent loop - and must never be compared with each other.
    """
    if not results:
        print("no episodes recorded - nothing to tabulate")
        return
    for model, task_cells in results.items():
        print("")
        print(f"=== {model} ===")
        _print_table(task_cells, "score: mean points (completed/recorded)", _cell_score)
        _print_table(task_cells, "mean tool calls",
                     lambda cell: _cell_number(cell, "mean_tool_calls"))
        _print_table(task_cells, "mean LLM calls",
                     lambda cell: _cell_number(cell, "mean_llm_calls"))
        _print_table(task_cells, "mean seconds per LLM request",
                     lambda cell: _cell_number(cell, "mean_llm_time_taken"))
        _print_table(task_cells, "mean seconds per tool call (server round trip)",
                     lambda cell: _cell_number(cell, "mean_time_taken"))
        _print_table(task_cells, "mean seconds per episode",
                     lambda cell: _cell_number(cell, "mean_episode_time"))


def write_csv(path, rows):
    """The long-format table (one row per episode) that pandas reads."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROW_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in ROW_FIELDS})
    return path


def write(args, trials):
    """Aggregate ``trials`` into ``_catalogue_*.json`` + ``*.csv`` and tabulate.

    ``trials`` is what was actually attempted (plus what ``--resume`` skipped), so
    the aggregate never claims episodes a ``--limit`` run did not reach. The
    aggregate itself sits in the batch folder, not in a model sub-folder: it spans
    every model of the batch. Returns ``(json_path, csv_path)``.
    """
    results, rows = collect(trials, args.log_folder, args.group_models)
    # `_smoke` rather than the backend name, so a scripted check can never
    # overwrite the aggregate of a real batch started with the same seed/tag.
    batch = "smoke" if args.smoke else args.backend
    stem = instructions.sanitise(f"_catalogue_{batch}_seed{args.seed}{args.tag}")
    json_path = os.path.join(args.log_folder, f"{stem}_summary.json")
    csv_path = os.path.join(args.log_folder, f"{stem}.csv")

    cells = []
    for model, task_cells in results.items():
        for task, level_cells in task_cells.items():
            for level, cell in level_cells.items():
                entry = OrderedDict([("model", model), ("task", task),
                                     ("level", level)])
                entry.update(cell)
                cells.append(entry)

    payload = OrderedDict([
        ("generated", datetime.datetime.now().isoformat(timespec="seconds")),
        ("catalogue", os.path.abspath(args.instructions_file)),
        ("backend", args.backend),
        ("models", args.models),
        ("seed", args.seed),
        ("repeats", args.repeats),
        ("preset_requested", args.preset),
        ("objective_map", args.objective_map),
        ("max_steps", args.max_steps),
        ("llm_timeout", args.llm_timeout),
        ("max_advance_ticks", args.max_advance_ticks),
        ("smoke", args.smoke),
        ("log_folder", args.log_folder),
        ("group_models", args.group_models),
        ("episodes", len(rows)),
        ("missing_episodes", sum(cell["missing"] for cell in cells)),
        ("results", results),
        ("cells", cells),
    ])
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    write_csv(csv_path, rows)
    print_tables(results)
    return json_path, csv_path

