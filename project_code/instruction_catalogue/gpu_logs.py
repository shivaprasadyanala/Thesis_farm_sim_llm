"""Where the GPU energy of a measured batch is kept, and in which shape.

``gpu_energy_cons.py`` wraps every episode of the catalogue in one NVML envelope;
this module owns the *files* that envelope produces, the way `results` owns the
score tables. A batch folder therefore answers "what did this cost" with the same
conventions as "what did it score"::

    <log-folder>/gpu_energy_logs/
        all_gpu_logs_per_episode_per_model.csv   wide:   one row per episode
        all_gpu_samples.csv                      long:   one row per NVML sample
        energy_by_cell.csv                       means per model x task x level
        baselines.json                           idle baselines, and when measured
        <model>/                                 one folder per model id
            episodes.csv                         this model's rows of the wide table
            <run>.json                           envelope + matrix cell + the row
            <run>.csv                            the NVML time series, as sampled

Why this shape
--------------
* **The series stays raw.** ``<run>.csv`` is a copy of the agent's own file
  (``unix_time_s, utc, energy_mj, power_w``), untouched - evidence a reviewer can
  re-integrate. The agent names that file after the run too (``/start?name=``), so
  ``~/alems-telemetry`` and the batch folder show the same name; a re-measured run
  is suffixed instead of overwriting the first attempt, and ``row["series"]`` - not
  a guessed file name - is what says which file belongs to which run.
* **A wide table is what statistics need.** ``all_gpu_logs_per_episode_per_model.csv``
  joins the energy of a run with its score, tokens, stop reason and verdict: one
  row per episode, every model, ready for pandas and directly comparable with the
  plain batch's ``_catalogue_*.csv``.
* **A long table is what plots need.** ``all_gpu_samples.csv`` holds one row per
  sample with ``elapsed_s`` and the cumulative ``energy_j`` since the run started,
  so "power draw over an episode" and "when did the spike happen" are one
  ``groupby`` away and never need the per-run files again.
* **Cells are what a chapter table needs.** ``energy_by_cell.csv`` averages a cell
  (model x task x level) over its phrasings and repeats with the spread next to the
  mean - a mean without its sd hides a 0.5x/2x split.
* **The per-run JSON is the source of truth.** It carries the row that appears in
  the tables, so every table is a *projection*: delete them and
  ``python -m instruction_catalogue.gpu_logs <log-folder>`` rebuilds them from the
  run files. No number is derived twice, and a half-written table cannot outlive
  the data it claims to summarise.

Where it lives
--------------
The tree sits inside the batch's ``--log-folder`` by default, because an experiment
folder should be one self-contained unit that can be archived, copied into the
thesis repo or re-analysed later. ``A_LEMS_GPU_LOGS`` overrides the root for the
case where the data belongs outside the batch folder (a shared ``telemetry/``
tree, a second disk); a relative value is resolved against the batch folder, so
``A_LEMS_GPU_LOGS=telemetry/gpu_energy_logs`` gives::

    <log-folder>/telemetry/gpu_energy_logs/<model>/<run>.csv

Per-model sub-folders use `instructions.model_folder`, the sanitising the batch
itself uses for its episode folders, so a model's energy files and its episode
logs always sit under the same name.

Units
-----
Joules, seconds, watts, as the agent reports them. ``net_energy_j`` is
``energy_j - baseline_w * duration_s`` and only exists when an idle baseline was
taken; a missing baseline is an empty cell, never a made-up zero.
"""

import argparse
import csv
import json
import os
import shutil
from collections import OrderedDict
from datetime import datetime, timezone
from statistics import mean, stdev

from . import instructions

#: Folder of the tree, inside the batch folder unless `ENV_ROOT` says otherwise.
DIRNAME = "gpu_energy_logs"

#: Environment variable naming an alternative root (absolute, or relative to the
#: batch folder). Read on every call: a batch can be pointed elsewhere per run.
ENV_ROOT = "A_LEMS_GPU_LOGS"

#: The cross-model table: one row per measured episode, every model, wide format.
EPISODES_CSV = "all_gpu_logs_per_episode_per_model.csv"

#: The tidy time series: one row per NVML sample, every episode, every model.
SAMPLES_CSV = "all_gpu_samples.csv"

#: Per-cell means (model x task x level), with the spread next to them.
CELLS_CSV = "energy_by_cell.csv"

#: The idle baselines ``net_energy_j`` was computed against, per GPU and moment.
BASELINES_JSON = "baselines.json"

#: Inside a model folder: the same wide table restricted to that model.
MODEL_EPISODES_CSV = "episodes.csv"

#: Order of the levels inside a cell table (the batch's own order).
LEVEL_ORDER = {"direct": 0, "indirect": 1, "implicit": 2}

#: Columns of ``all_gpu_logs_per_episode_per_model.csv`` (and of ``episodes.csv``).
#: Three blocks in this order: what was run (the matrix cell), how it ended (the
#: batch's verdict plus the episode's own numbers), what it cost (the envelope).
#: ``j_per_point``/``j_per_llm_call`` are the derived efficiencies a model
#: comparison is actually about, so they are materialised here once instead of
#: being recomputed - slightly differently - in every notebook.
EPISODE_FIELDS = (
    "run", "model", "task", "level", "prompt_index", "repeat", "seed", "preset",
    "objective",
    "exit_code", "stop_reason", "seconds", "episode_time", "points", "completed",
    "llm_calls", "tool_calls", "input_tokens", "output_tokens",
    "energy_j", "net_energy_j", "avg_power_w", "peak_power_w", "duration_s",
    "baseline_w", "method", "samples", "j_per_point", "j_per_llm_call",
    "gpu_run_id", "start_utc", "stop_utc",
    "series", "host_csv", "episode_log",
)

#: Columns of ``all_gpu_samples.csv``: the episode identity left, the sample right.
#: ``energy_j`` is cumulative since the run's first sample (counter difference, or
#: the trapezoidal integral when the GPU has no counter), so one column means
#: "joules since the run started" on every GPU.
SAMPLES_FIELDS = (
    "run", "model", "task", "level", "prompt_index", "repeat",
    "sample", "elapsed_s", "unix_time_s", "utc", "power_w", "energy_mj", "energy_j",
)

#: Columns of ``energy_by_cell.csv``.
CELL_FIELDS = (
    "model", "task", "level", "n", "n_ok",
    "energy_j_mean", "energy_j_sd", "energy_j_min", "energy_j_max",
    "net_energy_j_mean", "avg_power_w_mean", "peak_power_w_mean",
    "episode_time_mean", "points_mean", "completed_rate",
    "j_per_point_mean", "j_per_llm_call_mean",
)


# ---------------------------------------------------------------- the tree

def root(log_folder):
    """``<log-folder>/gpu_energy_logs``, or the `ENV_ROOT` override."""
    override = os.environ.get(ENV_ROOT)
    if override:
        if not os.path.isabs(override):
            override = os.path.join(log_folder, override)
        return os.path.abspath(override)
    return os.path.join(os.path.abspath(log_folder), DIRNAME)


def model_dir(log_folder, model):
    """The model's own folder in the tree (the batch's sanitised model id)."""
    return os.path.join(root(log_folder), instructions.model_folder(model))


def run_paths(log_folder, model, run):
    """``(json, csv)`` of one run: its record and its raw series."""
    folder = model_dir(log_folder, model)
    return os.path.join(folder, f"{run}.json"), os.path.join(folder, f"{run}.csv")


def label(path, base):
    """`path` relative to `base`, forward-slashed; the bare name if it escapes.

    Tables travel (thesis repo, a supervisor's machine), so a recorded path is
    relative to the batch folder whenever the file is inside it - the same rule
    `results` applies to its summary column.
    """
    try:
        relative = os.path.relpath(path, base)
    except ValueError:  # different drive on Windows
        return os.path.basename(path)
    if relative.startswith(os.pardir):
        return str(path).replace(os.sep, "/")
    return relative.replace(os.sep, "/")


def series_path(log_folder, row):
    """The series file of a recorded run, from the row's own ``series`` column.

    Not ``<run>.csv``: the file name is whatever the agent answered with (a
    re-measured run is suffixed), and older records were written when the agent
    still numbered its own files - the column is the only thing that says which
    file belongs to which run. ``None`` when the run has no series at all.
    """
    name = os.path.basename(str(row.get("series") or "").replace("\\", "/"))
    if not name:
        return None
    return os.path.join(model_dir(log_folder, row.get("model", "")), name)


# ---------------------------------------------------------------- numbers

def _number(text):
    """A float from a CSV/JSON cell, or ``None`` for an empty/unparsable one."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def _as_int(text):
    """An int for sorting (``"3"`` -> 3); unparsable sorts first."""
    try:
        return int(text)
    except (TypeError, ValueError):
        return 0


def _ratio(energy_j, per):
    """``energy / per`` rounded to 4 decimals, or ``""`` when that is meaningless."""
    energy, count = _number(energy_j), _number(per)
    if energy is None or not count or count <= 0:
        return ""
    return round(energy / count, 4)


def _mean(values, digits=3):
    """Mean over the numbers present (``None``/``""`` holes do not count)."""
    numbers = [value for value in values if isinstance(value, (int, float))]
    return round(mean(numbers), digits) if numbers else ""


def _spread(values, digits=3):
    """Standard deviation over the numbers present; ``""`` below two samples."""
    numbers = [value for value in values if isinstance(value, (int, float))]
    return round(stdev(numbers), digits) if len(numbers) > 1 else ""


# ---------------------------------------------------------------- the series

def series_rows(path):
    """Parse one time series into per-sample rows, plus the number of bad lines.

    The agent writes ``unix_time_s, utc, energy_mj, power_w`` and ``energy_mj`` is
    the *cumulative* hardware counter, so the physics is in the differences. Each
    row therefore gets ``energy_j``: the counter difference since the first sample
    when the GPU has a counter, otherwise the trapezoidal integral of ``power_w`` -
    one column, one meaning, whichever method the host used.

    Integration uses the wall clock because the agent logs no monotonic one; a
    clock step during a run then shows up as an odd ``elapsed_s`` in this table
    rather than being silently absorbed.
    """
    try:
        handle = open(path, "r", encoding="utf-8", newline="")
    except OSError as exc:
        print(f"note: cannot read the GPU series {path} ({exc})")
        return [], 1

    rows, skipped = [], 0
    first_time = first_mj = None
    prev_time = prev_power = None
    integrated = 0.0
    with handle:
        for record in csv.DictReader(handle):
            wall = _number(record.get("unix_time_s"))
            power = _number(record.get("power_w"))
            if wall is None or power is None:
                skipped += 1
                continue
            counter = _number(record.get("energy_mj"))
            if first_time is None:
                first_time, first_mj = wall, counter
            if counter is not None and first_mj is not None:
                energy_j = (counter - first_mj) / 1000.0
            else:
                if prev_time is not None:
                    integrated += 0.5 * (power + prev_power) * (wall - prev_time)
                energy_j = integrated
            prev_time, prev_power = wall, power
            rows.append(OrderedDict((
                ("sample", len(rows) + 1),
                ("elapsed_s", round(wall - first_time, 3)),
                ("unix_time_s", record.get("unix_time_s", "")),
                ("utc", record.get("utc", "")),
                ("power_w", round(power, 3)),
                ("energy_mj", "" if counter is None else record.get("energy_mj", "")),
                ("energy_j", round(energy_j, 3)),
            )))
    return rows, skipped


def series_stats(path):
    """``(samples, peak_power_w, energy_j)`` of a series, or zeros when unreadable.

    Two of these are data-quality numbers as much as physics: a run whose sample
    count is far below ``duration_s / A_LEMS_SAMPLE_MS`` was not sampled densely
    enough to trust its average, and the series' ``energy_j`` can be compared with
    the envelope's to catch a counter reset.
    """
    rows, _ = series_rows(path)
    if not rows:
        return 0, "", ""
    return len(rows), max(row["power_w"] for row in rows), rows[-1]["energy_j"]


def series_name(run, source=None):
    """The file name a run's series gets in its model folder.

    The agent names its file after the run, so the copy normally keeps that name and
    the tree and the mount agree. Two cases fall back to ``<run>.csv``: an older
    agent, whose file is still called ``gpu0_<UTC>``, and a re-measured run, which
    the agent had to suffix (``<run>_2``) to keep both attempts - the suffix is part
    of the name it answers with, so it is kept here too, and neither attempt's series
    is silently overwritten.
    """
    stem = os.path.splitext(os.path.basename(str(source or "")))[0]
    if stem and stem not in (".", "..") and (stem == run or stem.startswith(run + "_")):
        return stem + ".csv"
    return os.path.basename(str(run)) + ".csv"


def copy_series(source, target):
    """Copy the agent's CSV to the run's own name; a missing mount is not fatal.

    The mount (``GPU_TELEMETRY_DIR``) is the host's evidence and may not be there
    when this driver runs on the host itself; the copy then fails with a note and
    the measurement - which is already in hand - is kept without its series.
    """
    try:
        shutil.copyfile(source, target)
        return target
    except OSError as exc:
        print(f"      gpu: no series copy ({exc}); the host file is {source}")
        return None


def record(log_folder, model, run, envelope, row, source=None):
    """Write one measured run into the tree; returns the row as recorded.

    ``envelope`` is what the driver got from ``/start`` and ``/stop``, ``row`` the
    episode row it already built, ``source`` the agent's CSV. The written
    ``<run>.json`` carries `row`, so the tables are pure projections of the run
    files and a re-run of one episode replaces exactly one row of every table.

    The copy happens before the row is finished because the series adds three
    columns of its own (``samples``, ``peak_power_w`` and the cross-check between
    the two energy figures); an unreadable series leaves them empty instead of
    losing the energy that is already known.
    """
    folder = model_dir(log_folder, model)
    os.makedirs(folder, exist_ok=True)
    json_path, _ = run_paths(log_folder, model, run)

    row = dict(row)
    row["run"] = run
    row["series"] = ""
    if source:
        target = os.path.join(folder, series_name(run, source))
        copied = copy_series(source, target)
        if copied:
            row["series"] = label(copied, log_folder)
    if row["series"]:
        samples, peak, integrated = series_stats(
            os.path.join(log_folder, row["series"]))
        row["samples"] = samples
        row["peak_power_w"] = peak
        measured = _number(row.get("energy_j"))
        if measured and integrated != "" and \
                abs(measured - integrated) > 0.05 * measured:
            print(f"      gpu: series says {integrated} J, envelope says "
                  f"{measured} J - check the counter or the sampling")
    else:
        row["samples"] = row.get("samples", "")
        row["peak_power_w"] = row.get("peak_power_w", "")
    row["j_per_point"] = _ratio(row.get("energy_j"), row.get("points"))
    row["j_per_llm_call"] = _ratio(row.get("energy_j"), row.get("llm_calls"))

    payload = OrderedDict((
        ("run", run),
        ("row", OrderedDict((field, row.get(field, "")) for field in EPISODE_FIELDS)),
        ("start", envelope.get("start") or {}),
        ("stop", envelope.get("stop") or {}),
    ))
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return row


# ---------------------------------------------------------------- reading back

def _order(row):
    """Matrix order: model -> task -> level -> phrasing -> repeat."""
    level = str(row.get("level", ""))
    return (str(row.get("model", "")), str(row.get("task", "")),
            LEVEL_ORDER.get(level, 9), level,
            _as_int(row.get("prompt_index")), _as_int(row.get("repeat")))


def episode_rows(log_folder):
    """Every recorded episode row, in matrix order.

    Read from the per-run JSONs, which are the source of truth: a table can be
    deleted, another model's folder can arrive late, and the answer is still the
    union of the runs that measured something. A corrupt record is reported and
    skipped - one bad file must not make the whole batch unreadable.
    """
    rows = []
    base = root(log_folder)
    if not os.path.isdir(base):
        return rows
    for folder in sorted(os.listdir(base)):
        path = os.path.join(base, folder)
        if not os.path.isdir(path):
            continue
        for name in sorted(os.listdir(path)):
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(path, name), "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
            except (OSError, ValueError) as exc:
                print(f"note: cannot read {os.path.join(path, name)} ({exc})")
                continue
            row = payload.get("row") if isinstance(payload, dict) else None
            if isinstance(row, dict) and row.get("run"):
                rows.append(row)
    rows.sort(key=_order)
    return rows


def measured(log_folder):
    """``{run: row}`` of the runs that already have energy - what ``--resume`` skips.

    A row counts only when the envelope produced a number *and* the episode passed
    the batch's verdict: an episode can own the artefacts of a plain batch run and
    no energy at all, and re-running it is the only way to get its measurement.
    """
    return {row["run"]: row for row in episode_rows(log_folder)
            if row.get("energy_j") not in ("", None)
            and str(row.get("exit_code")) == "0"}


def cell_rows(rows):
    """``energy_by_cell.csv``: a cell (model x task x level) averaged over its runs.

    Means carry their standard deviation, because a phrasing that costs 0.5x and
    one that costs 2x average to a number that describes neither. ``n`` counts the
    attempts and ``n_ok`` the ones that passed the verdict, so a cell built from
    failures is recognisable instead of merely cheap.
    """
    cells = OrderedDict()
    for row in rows:
        key = (str(row.get("model", "")), str(row.get("task", "")),
               str(row.get("level", "")))
        cells.setdefault(key, []).append(row)

    out = []
    for (model, task, level), group in cells.items():
        energy = [value for value in
                  (_number(row.get("energy_j")) for row in group)
                  if value is not None]
        out.append(OrderedDict((
            ("model", model), ("task", task), ("level", level),
            ("n", len(group)),
            ("n_ok", sum(1 for row in group if str(row.get("exit_code")) == "0")),
            ("energy_j_mean", _mean(energy)),
            ("energy_j_sd", _spread(energy)),
            ("energy_j_min", round(min(energy), 3) if energy else ""),
            ("energy_j_max", round(max(energy), 3) if energy else ""),
            ("net_energy_j_mean",
             _mean([_number(row.get("net_energy_j")) for row in group])),
            ("avg_power_w_mean",
             _mean([_number(row.get("avg_power_w")) for row in group])),
            ("peak_power_w_mean",
             _mean([_number(row.get("peak_power_w")) for row in group])),
            ("episode_time_mean",
             _mean([_number(row.get("episode_time")) for row in group])),
            ("points_mean", _mean([_number(row.get("points")) for row in group])),
            ("completed_rate",
             round(sum(1 for row in group if row.get("completed") is True)
                   / len(group), 3)),
            ("j_per_point_mean",
             _mean([_number(row.get("j_per_point")) for row in group], 4)),
            ("j_per_llm_call_mean",
             _mean([_number(row.get("j_per_llm_call")) for row in group], 4)),
        )))
    return out


def _write_csv(path, fields, rows):
    """Rewrite one CSV from scratch: a table never survives its own data."""
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields),
                                extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def write_tables(log_folder):
    """Rewrite the wide table, the per-model copies and the cell table.

    Called after *every* measured episode, so an interrupted batch leaves tables
    that match the runs on disk instead of a header plus whatever was flushed last.
    Returns ``{name: path}`` for the console.
    """
    base = root(log_folder)
    os.makedirs(base, exist_ok=True)
    rows = episode_rows(log_folder)

    paths = {"episodes": _write_csv(os.path.join(base, EPISODES_CSV),
                                    EPISODE_FIELDS, rows),
             "cells": _write_csv(os.path.join(base, CELLS_CSV),
                                 CELL_FIELDS, cell_rows(rows))}

    folders = OrderedDict()
    for row in rows:
        folders.setdefault(str(row.get("model", "")), []).append(row)
    written = []
    for model, group in folders.items():
        folder = model_dir(log_folder, model)
        os.makedirs(folder, exist_ok=True)
        written.append(_write_csv(os.path.join(folder, MODEL_EPISODES_CSV),
                                  EPISODE_FIELDS, group))
    paths["models"] = written
    return paths


def write_samples(log_folder):
    """Rewrite ``all_gpu_samples.csv``: every sample of every recorded run.

    Built once at the end of a batch rather than per episode: it is the only table
    that grows with the sampling rate instead of with the matrix, and it is a pure
    projection of the per-run series, so deleting it costs nothing but this call.
    Returns ``(path, rows, skipped)``.
    """
    base = root(log_folder)
    os.makedirs(base, exist_ok=True)
    out = []
    skipped = 0
    for run_row in episode_rows(log_folder):
        series = series_path(log_folder, run_row)
        if not series or not os.path.isfile(series):
            continue
        samples, bad = series_rows(series)
        skipped += bad
        identity = {field: run_row.get(field, "") for field in
                    ("run", "model", "task", "level", "prompt_index", "repeat")}
        for sample in samples:
            out.append(OrderedDict(list(identity.items()) + list(sample.items())))
    path = _write_csv(os.path.join(base, SAMPLES_CSV), SAMPLES_FIELDS, out)
    return path, len(out), skipped


# ---------------------------------------------------------------- baselines

def write_baseline(log_folder, reply, gpu=None, seconds=None):
    """Append one idle baseline to ``baselines.json`` - provenance for net energy.

    ``net_energy_j`` is only as trustworthy as the idle power subtracted from it,
    so the value, the GPU and the moment are kept next to the runs it applies to
    instead of living only in the console output of one batch.
    """
    base = root(log_folder)
    os.makedirs(base, exist_ok=True)
    path = os.path.join(base, BASELINES_JSON)
    entry = OrderedDict((
        ("utc", datetime.now(timezone.utc).isoformat(timespec="seconds")),
        ("gpu", reply.get("gpu", gpu)),
        ("baseline_w", reply.get("baseline_w")),
        ("seconds", reply.get("seconds", seconds)),
    ))
    recorded = []
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                recorded = json.load(handle).get("baselines") or []
        except (OSError, ValueError) as exc:
            print(f"note: cannot read {path} ({exc}); starting a new file")
    recorded.append(entry)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"baselines": recorded}, handle, indent=2)
    return path


# ---------------------------------------------------------------- the console

def describe(log_folder):
    """``{"episodes", "models", "series", "baselines"}`` of the tree, for the console.

    Cheap enough to print at the end of a batch: it answers "how much evidence is
    actually on disk" without opening a single series file.
    """
    rows = episode_rows(log_folder)
    base = root(log_folder)
    series = 0
    for row in rows:
        path = series_path(log_folder, row)
        if path and os.path.isfile(path):
            series += 1
    models = OrderedDict()
    for row in rows:
        model = str(row.get("model", ""))
        models[model] = models.get(model, 0) + 1
    baselines = 0
    path = os.path.join(base, BASELINES_JSON)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                baselines = len(json.load(handle).get("baselines") or [])
        except (OSError, ValueError):
            baselines = 0
    return {"root": base, "episodes": len(rows), "series": series,
            "models": models, "baselines": baselines}


def main(argv=None):
    """Rebuild the tables of an existing batch folder.

    Nothing here measures anything: it re-projects the per-run JSONs after the
    tables were deleted, a model folder arrived from another machine, or a run was
    removed by hand. ``python -m instruction_catalogue.gpu_logs <log-folder>``.
    """
    parser = argparse.ArgumentParser(
        prog="python -m instruction_catalogue.gpu_logs",
        description="Rebuild the GPU energy tables of a measured batch folder from "
                    "its per-run records (the <model>/<run>.json files).")
    parser.add_argument("log_folder", nargs="?", default=".",
                        help="the batch folder, i.e. the --log-folder of the run")
    parser.add_argument("--no-samples", action="store_true",
                        help=f"skip {SAMPLES_CSV} (the only table that grows "
                             f"with the sampling rate)")
    args = parser.parse_args(argv)

    log_folder = os.path.abspath(args.log_folder)
    rows = episode_rows(log_folder)
    if not rows:
        print(f"nothing to rebuild: no run record under {root(log_folder)}")
        return 1
    paths = write_tables(log_folder)
    print(f"episodes: {len(rows)} run(s) -> {paths['episodes']}")
    print(f"cells:    {paths['cells']}")
    for path in paths["models"]:
        print(f"model:    {path}")
    if not args.no_samples:
        path, count, skipped = write_samples(log_folder)
        print(f"samples:  {path} ({count} row(s)"
              + (f", {skipped} unreadable line(s)" if skipped else "") + ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
