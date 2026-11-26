"""One GPU energy envelope per instruction-catalogue episode.

``run_instruction_catalogue.py`` answers *what a model does* with each phrasing of
the catalogue. This script answers *what that costs*: the NVML agent on the host
(``host_gpu_agent_nvml.py``) is started before every single episode and stopped
again right after it::

    POST /start?gpu=0&name=<run>   -> run id, method, path of the CSV series
    <one episode subprocess>
    POST /stop?gpu=0               -> energy_j, avg_power_w, duration_s, method,
                                      baseline_w and net_energy_j when known

One envelope per episode is the whole point. An envelope around the batch would
only report the energy of "300 episodes", in which a cheap model and an expensive
one are indistinguishable; per episode the energy joins the score, the stop reason
and the token counts of exactly that wording.

Nothing of the batch is re-implemented: the flags come from
`instruction_catalogue.cli`, the matrix from `instruction_catalogue.instructions`,
the command line of an episode from `runner.build_command`, and the verdict from
the same ``_summary.json`` and `results.FAILED_STOP_REASONS` the plain batch uses.
A measured episode is therefore the byte-identical episode of the plain batch -
the only difference is the envelope around it, which is exactly what makes the two
tables comparable.

Run names
---------
A run is named by `Trial.artefact_stem()` - the very stem the episode itself uses
for its ``.log``/``_summary.json``::

    <model>_<preset>_seed<seed>_<task>_<level>_p<n>_r<n>

The name travels to the agent as ``/start?gpu=0&name=<run>``, so the host's
evidence (``~/alems-telemetry/<run>.csv`` and ``.json``) carries the same name as
the episode it measured, and the copy in the batch folder is a same-name duplicate
rather than a rename. The agent answers with the name it actually used - a
re-measured episode gets ``<run>_2`` instead of overwriting the first attempt - and
everything downstream keys off that answer, never off an assumed file name, so an
older agent that ignores ``name`` still works (its files keep the timestamped
``gpu<N>_<UTC>`` names).

What lands on disk
------------------
`instruction_catalogue.gpu_logs` owns the layout, so a measured batch folder
carries its energy evidence next to its score tables, grouped by model::

    <log-folder>/gpu_energy_logs/
        all_gpu_logs_per_episode_per_model.csv   one row per episode, every model
        all_gpu_samples.csv                      one row per NVML sample
        energy_by_cell.csv                       mean/sd per model x task x level
        baselines.json                           the idle baselines, and when taken
        <model>/
            episodes.csv                         that model's rows only
            <run>.json                           envelope + matrix cell + the row
            <run>.csv                            the time series, copied from the mount

The wide table is the one to read: it joins the energy of a run with its points,
tokens, stop reason and verdict, so "J per point" per model is a column, not a
notebook. The long table is the one to plot: per-sample power and cumulative
joules, every run in one file. The per-run ``.json`` is the source of truth - the
tables are projections of it and can be rebuilt at any time with
``python -m instruction_catalogue.gpu_logs <log-folder>``. ``A_LEMS_GPU_LOGS``
points the tree somewhere else (a shared ``telemetry/`` tree) if the batch folder
is not where the data should live.

The host keeps its own timestamped evidence untouched; the copy is what gives the
batch folder the name the rest of the experiment is indexed by. Once the loop is
done, the batch's own aggregate of exactly the episodes that were attempted is
written too (``results.write``), so one folder holds the score table, its energy
table and the raw series behind both.

Requirements
------------
The agent must run on the machine with the GPU and be reachable from here, and its
log folder must be mounted for the CSV copies::

    A_LEMS_TOKEN=... python3 host_gpu_agent_nvml.py
    docker run -v ~/alems-telemetry:/telemetry ...

Environment:

    A_LEMS_TOKEN         shared secret of the agent (required)
    GPU_TELEMETRY_URL    agent base URL (default http://host.docker.internal:8765)
    A_LEMS_GPU           GPU index to measure (default 0)
    GPU_TELEMETRY_DIR    mount point of ~/alems-telemetry (default /telemetry)
    A_LEMS_BASELINE_S    idle baseline before the matrix, 0 = off (default 10)
    A_LEMS_GPU_LOGS      root of the log tree (default <log-folder>/gpu_energy_logs)

Usage (from ``project_code``, like the batch it wraps)::

    python gpu_energy_cons.py --dry-run
    python gpu_energy_cons.py --vllm --levels all --repeats 3
    python gpu_energy_cons.py --model all --limit 2 --resume

Exit codes are the batch's - 0 every attempt ok, 1 an episode failed, 2 invalid
matrix, 130 interrupted - plus 3 for "the collector cannot measure": then no
episode is run at all, because an episode that runs unmeasured is a hole in the
table this script exists to fill.
"""

import contextlib
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

from farm_agent import config as farm_config
from instruction_catalogue import cli, gpu_logs, instructions, results, runner

# ---------------------------------------------------------------- settings

#: Where the agent listens: the host as seen from the dev container. A driver
#: started on the host itself passes GPU_TELEMETRY_URL=http://127.0.0.1:8765.
HOST = os.environ.get("GPU_TELEMETRY_URL",
                      "http://host.docker.internal:8765").rstrip("/")

#: The GPU the envelope is taken on (the agent's own variable, same default).
GPU = int(os.environ.get("A_LEMS_GPU", "0"))

#: Where ``~/alems-telemetry`` of the host is mounted here (empty = not mounted).
TELEMETRY_DIR = os.environ.get("GPU_TELEMETRY_DIR", "/telemetry")

#: Idle baseline of the GPU before the matrix, in seconds (0 turns it off).
BASELINE_S = float(os.environ.get("A_LEMS_BASELINE_S", "10"))

#: One failed connection may be a restarting agent; three in a row is a dead host.
ATTEMPTS = 3
RETRY_DELAY = 5.0

#: ``/stop`` answers after the last sample and ``/baseline`` sleeps its seconds.
TIMEOUT = 60.0


class CollectorError(RuntimeError):
    """The agent refused or cannot be reached: this episode cannot be measured."""


# ---------------------------------------------------------------- the agent


def token():
    """``A_LEMS_TOKEN`` - read lazily, so ``--help``/``--dry-run`` need no secret."""
    value = os.environ.get("A_LEMS_TOKEN")
    if not value:
        raise CollectorError("A_LEMS_TOKEN is not set - the agent requires it "
                             "(export the value the host agent was started with)")
    return value


def _error_text(payload):
    """The agent's own ``{"error": ...}`` message, or whatever the body was."""
    try:
        message = json.loads(payload).get("error")
    except (ValueError, AttributeError):
        message = None
    return str(message or payload or "(empty body)")


def _request(path, query=None, method="POST"):
    """One authenticated call to the agent, returned as the decoded JSON reply.

    Only connection failures are retried: a 401/404/409/500 is deterministic and
    its ``error`` field is the whole diagnosis, so a retry would just delay it.
    """
    url = f"{HOST}{path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)
    headers = {"Authorization": f"Bearer {token()}"}
    body = b"" if method == "POST" else None
    failure = None
    for attempt in range(1, ATTEMPTS + 1):
        request = urllib.request.Request(url, data=body, headers=headers,
                                         method=method)
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
                payload = response.read().decode("utf-8", "replace")
            break
        except urllib.error.HTTPError as exc:
            try:
                detail = exc.read().decode("utf-8", "replace")
            except OSError:
                detail = ""
            raise CollectorError(f"{path} -> HTTP {exc.code}: "
                                 f"{_error_text(detail)}")
        except (urllib.error.URLError, OSError) as exc:
            failure = exc
            if attempt < ATTEMPTS:
                print(f"      gpu: {path} failed ({exc}); retrying "
                      f"({attempt}/{ATTEMPTS - 1}) in {RETRY_DELAY:.0f}s")
                time.sleep(RETRY_DELAY)
    else:
        raise CollectorError(f"cannot reach the GPU agent at {url}: {failure}")
    try:
        return json.loads(payload)
    except ValueError:
        raise CollectorError(f"{path} answered without JSON: {payload[:200]!r}")


def gpu_collector(action, seconds=None, name=None):
    """``status``, ``start``, ``stop`` or ``baseline`` on `GPU`, as a dict.

    Mirrors the agent's endpoints one to one. ``name`` is the run's name and only
    means anything to ``start``: the agent writes its CSV and JSON under that name,
    so the host's evidence and the batch folder agree (see the module docstring).
    """
    if action == "status":
        return _request("/status", method="GET")
    query = {"gpu": GPU}
    if action == "start" and name:
        query["name"] = name
    if action == "baseline":
        query["seconds"] = int(seconds if seconds is not None else BASELINE_S)
    return _request(f"/{action}", query)


@contextlib.contextmanager
def measuring(run):
    """The envelope of one run: start it here, always stop it again.

    ``run`` is the episode's own name and goes to the agent, which names its CSV
    and JSON with it - so an older agent that ignores the parameter still answers
    with a usable path, and the ``run_id`` it answers with is what this side uses.

    Yields a dict that receives the ``/stop`` summary, so the caller can log the
    energy of exactly this episode. ``stop`` sits in ``finally`` because a crashing
    or hanging episode must never leave the host sampling: an open run would also
    make the next ``/start`` answer 409 and take the rest of the matrix with it.
    """
    started = gpu_collector("start", name=run)
    if not str(started.get("run_id", "")).startswith(run):
        print(f"      gpu: the agent kept its own name ({started.get('run_id')}); "
              f"update host_gpu_agent_nvml.py to name the series after the run")
    print(f"      gpu: measuring {started.get('run_id')} on GPU "
          f"{started.get('gpu')} ({started.get('gpu_name')}, "
          f"{started.get('method')})")
    envelope = {"run": run, "start": started, "stop": None}
    try:
        yield envelope
    finally:
        try:
            envelope["stop"] = gpu_collector("stop")
            print(f"      gpu: stopped {envelope['stop'].get('run_id')}: "
                  f"{envelope['stop'].get('energy_j')} J in "
                  f"{envelope['stop'].get('duration_s')} s")
        except CollectorError as exc:
            print(f"      gpu: cannot stop the run ({exc})")
            print(f"      gpu: it may still be sampling - POST "
                  f"{HOST}/stop?gpu={GPU} on the host before the next run")


def collector_preflight():
    """``GET /status``: token valid, GPU there, and no foreign run still open."""
    try:
        health = gpu_collector("status")
    except CollectorError as exc:
        print(f"error: the GPU agent is not usable ({exc})")
        print("       start it on the host: A_LEMS_TOKEN=... python3 "
              "host_gpu_agent_nvml.py")
        return False
    gpus = health.get("gpus") or []
    if not gpus:
        print("error: the GPU agent reports no GPU at all")
        return False
    for gpu in gpus:
        baseline = gpu.get("baseline_w")
        counter = "yes" if gpu.get("energy_counter") else "no (integrated power)"
        print(f"gpu {gpu.get('index')}: {gpu.get('name')} | energy counter "
              f"{counter} | baseline "
              f"{baseline if baseline is not None else '-'} W | active run "
              f"{gpu.get('active_run') or '-'}")
    target = next((gpu for gpu in gpus if gpu.get("index") == GPU), None)
    if target is None:
        print(f"error: GPU {GPU} does not exist on the host (it reports "
              f"{[gpu.get('index') for gpu in gpus]})")
        return False
    if target.get("active_run"):
        print(f"error: a run is already active on GPU {GPU} "
              f"({target['active_run']}); /start would answer 409, and stopping a "
              f"run this script did not open would corrupt its energy - finish it "
              f"on the host first")
        return False
    return True


def measure_baseline():
    """Idle power once before the matrix, so every ``/stop`` also reports net energy.

    Optional by design: the agent refuses while the GPU is busy (409), and a
    missing baseline only costs accuracy - it must never cost the batch its runs.
    Returns the agent's reply (or ``None``) so the value is also written to
    ``baselines.json``, where the runs it applies to can find it later.
    """
    if BASELINE_S <= 0:
        print("baseline: off (A_LEMS_BASELINE_S=0) - energy stays gross draw")
        return None
    try:
        reply = gpu_collector("baseline")
    except CollectorError as exc:
        print(f"note: no idle baseline ({exc}); energy stays gross draw")
        return None
    print(f"baseline: {reply.get('baseline_w')} W idle over "
          f"{reply.get('seconds')} s - every /stop now also reports net_energy_j")
    return reply


def telemetry_path(host_csv):
    """The agent's CSV as seen from here: the mount, not the host's own path.

    The agent reports an absolute path on the host (``/home/<user>/alems-telemetry/
    <run>.csv``, or ``gpu0_<UTC>.csv`` for an agent that does not take a run name);
    inside the container the same file is ``<GPU_TELEMETRY_DIR>/<the same name>``.
    The mounted copy is preferred and the reported path is the fallback, so a driver
    started on the host itself needs no mount at all - and a wrong mount is reported
    with the path it tried.
    """
    if not host_csv:
        return None
    mounted = os.path.join(TELEMETRY_DIR, os.path.basename(host_csv))
    if os.path.isfile(mounted):
        return mounted
    return host_csv if os.path.isfile(host_csv) else mounted


def episode_fields(episode):
    """The episode's own numbers as columns of the energy table.

    Taken from the same ``_summary.json`` the batch's aggregate reads, and with the
    same definitions (`results.read`: ``metrics.objective_completed`` is what
    "completed" means, not the objective's own flag), so the energy table and the
    results table can never disagree about a run's score or token count - the point
    of measuring here rather than in a second harness.
    """
    metrics = episode.get("metrics") or {}
    tool_calls = episode.get("no_of_tool_calls")
    if not tool_calls:
        # `--check-api` episodes never append to `time_taken`, so the action list is
        # the fallback that keeps a smoke row from looking like an empty episode.
        tool_calls = len(episode.get("tool_calls") or [])
    return {"episode_time": episode.get("episode_time", ""),
            "points": episode.get("points_gained", ""),
            "completed": bool(metrics.get("objective_completed")),
            "llm_calls": episode.get("no_of_llm_calls", ""),
            "tool_calls": tool_calls,
            "input_tokens": episode.get("total_input_tokens", ""),
            "output_tokens": episode.get("total_output_tokens", "")}


def build_args(argv=None):
    """argv with the defaults of a measurement batch filled in.

    Two defaults differ from the plain batch on purpose:

    * ``--model all``: measuring one model is a smoke test, measuring the registry
      is the experiment;
    * ``--preset normal``: the energy of one model is only comparable to another in
      the same world, so the per-task easy/normal/hard mapping (``--preset
      default``) is the wrong default here.

    ``--smoke`` (takes no model) and the ``--list-*`` previews (they describe the
    plain batch) get nothing injected.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(arg in ("--smoke", "--list-models", "--list-tasks") for arg in argv):
        return argv
    if not any(arg == "--model" or arg.startswith("--model=") for arg in argv):
        argv += ["--model", "all"]
    if not any(arg == "--preset" or arg.startswith("--preset=") for arg in argv):
        argv += ["--preset", "normal"]
    return argv


# ---------------------------------------------------------------- the batch


def _run_episode(args, trial):
    """One episode, exactly as `runner.run` runs it; returns the exit code."""
    try:
        return subprocess.run(runner.build_command(args, trial),
                              cwd=runner.PROJECT_CODE,
                              timeout=args.trial_timeout or None).returncode
    except subprocess.TimeoutExpired:
        print(f"      episode exceeded --trial-timeout {args.trial_timeout:.0f}s")
        return -1
    except OSError as exc:
        print(f"      cannot start the episode: {exc}")
        return -1


def measure_one(args, trial):
    """One episode inside its envelope; returns its row of the energy table.

    The verdict is the batch's (exit 0, a readable ``_summary.json``, a stop reason
    that is not an infrastructure failure), and two energy verdicts are added: a
    refused ``/start`` fails the run without running it (code 3), and an envelope
    that never closed cannot produce energy (code -4).

    Every outcome is recorded, failures included: a table that only lists the runs
    that worked cannot be joined with the aggregate, and a hole is information.
    """
    run = trial.artefact_stem()
    row = {
        "run": run, "model": trial.model, "task": trial.task, "level": trial.level,
        "prompt_index": trial.prompt_index, "repeat": trial.repeat,
        "seed": trial.seed, "preset": trial.preset, "objective": trial.objective,
        "seconds": "", "exit_code": "", "stop_reason": "",
        "episode_log": gpu_logs.label(
            os.path.join(trial.folder(args.log_folder, args.group_models),
                         f"{run}.log"), args.log_folder),
    }
    started = time.time()
    try:
        with measuring(run) as envelope:
            code = _run_episode(args, trial)
    except CollectorError as exc:
        print(f"      {exc}")
        print("      the episode was not run: an unmeasured episode is a hole in "
              "the table")
        row["exit_code"] = 3
        row["seconds"] = round(time.time() - started, 1)
        return gpu_logs.record(args.log_folder, trial.model, run,
                               {"start": None, "stop": None}, row)
    row["seconds"] = round(time.time() - started, 1)

    episode = results.load_summary(
        trial.summary_path(args.log_folder, args.group_models))
    if episode is not None:
        row.update(episode_fields(episode))
    stop_reason = (episode or {}).get("stop_reason") or ""
    row["stop_reason"] = stop_reason
    if code == 0:
        if episode is None:
            # A zero exit without the artefact would make --resume re-run the
            # episode forever and hide it from the aggregate.
            code = -2
            print("      exited 0 but wrote no _summary.json - counting it as a "
                  "failure")
        elif stop_reason in results.FAILED_STOP_REASONS:
            code = -3
            print(f"      exited 0 but the episode stopped with {stop_reason!r}")

    stop = envelope.get("stop") or {}
    if stop:
        row.update(energy_j=stop.get("energy_j", ""),
                   net_energy_j=stop.get("net_energy_j", ""),
                   avg_power_w=stop.get("avg_power_w", ""),
                   duration_s=stop.get("duration_s", ""),
                   baseline_w=stop.get("baseline_w", ""),
                   method=stop.get("method", ""),
                   gpu_run_id=stop.get("run_id", ""),
                   start_utc=stop.get("start_utc", ""),
                   stop_utc=stop.get("stop_utc", ""),
                   host_csv=str(stop.get("csv", "")).replace(os.sep, "/"))
    elif code == 0:
        # The artefacts exist, but the run has no energy - which is the one number
        # this script exists to produce.
        code = -4
        print("      the envelope never closed with a summary - no energy for "
              "this run")
    row["exit_code"] = code

    row = gpu_logs.record(args.log_folder, trial.model, run, envelope, row,
                          source=telemetry_path(stop.get("csv")))
    if stop:
        net = row.get("net_energy_j")
        print(f"      energy: {row['energy_j']} J ({row['method']}), "
              f"{row['avg_power_w']} W mean"
              + (f", {net} J above idle" if net not in ("", None) else "")
              + (f", {row['samples']} sample(s), peak {row['peak_power_w']} W"
                 if row.get("samples") else ""))
    return row


# ---------------------------------------------------------------- entry point


def main(argv=None):
    """Resolve the matrix, check the agent, then one envelope per episode."""
    os.chdir(runner.PROJECT_CODE)   # the cwd of an episode: relative paths agree
    farm_config.load_config()
    raw = build_args(argv)
    if any(arg in ("-h", "--help") for arg in raw):
        print("gpu_energy_cons.py: the flags below are the batch's - every episode "
              "runs inside a GPU collector envelope")
        print("  collector: GPU_TELEMETRY_URL, A_LEMS_TOKEN, A_LEMS_GPU, "
              "GPU_TELEMETRY_DIR, A_LEMS_BASELINE_S")
        print("  logs: A_LEMS_GPU_LOGS (root of the energy tree, default "
              "<log-folder>/gpu_energy_logs)")
        print("  extra exit code: 3 = the collector cannot measure; no episode is "
              "run then")
        print("  full contract: module docstring of gpu_energy_cons.py")
        print("")
    args = cli.parse_args(raw)

    try:
        catalogue = instructions.load_catalogue(args.instructions_file)
        groups = instructions.resolve_task_groups(catalogue, args.tasks)
        levels = instructions.resolve_levels(args.levels)
        if args.list_models or args.list_tasks:
            # The previews are the plain batch's: no episode, no envelope.
            return cli.main(raw)
        if args.smoke:
            args.models = ["check_api"]
            trials = instructions.smoke_trials(catalogue, groups, args.preset,
                                               args.objective_map, args.seed)
        else:
            if str(args.model).strip().lower() == "all":
                args.models = cli.models_for_backend(
                    cli.read_models(args.models_file), args.backend,
                    args.models_file)
            else:
                args.models = [args.model]
            trials = instructions.expand_trials(
                catalogue, args.models, args.backend, groups, levels, args.preset,
                args.objective_map, args.seed, args.repeats)
        if args.group_models:
            # Two model ids that sanitise alike would share one folder.
            instructions.model_folders(args.models)
    except instructions.CatalogueError as exc:
        print(f"error: {exc}")
        return 2

    if args.dry_run:
        return instructions.print_matrix(trials, args)

    args.log_folder = os.path.abspath(args.log_folder)
    os.makedirs(args.log_folder, exist_ok=True)
    # The energy tree, and what an earlier invocation already measured: `--resume`
    # trusts the tree's own records, not `_summary.json`, because a plain batch run
    # can own an episode's artefacts and no energy at all.
    logs = gpu_logs.root(args.log_folder)
    os.makedirs(logs, exist_ok=True)
    previous = gpu_logs.measured(args.log_folder)

    baseline_note = f"{BASELINE_S:.0f}s" if BASELINE_S > 0 else "off"
    print(f"collector: {HOST} | GPU {GPU} | idle baseline {baseline_note}")
    print(f"gpu logs:  {logs}")
    print("")

    code = runner.preflight(args, trials)
    if code:
        return code
    if not collector_preflight():
        return 3
    baseline = measure_baseline()
    if baseline:
        gpu_logs.write_baseline(args.log_folder, baseline)

    summary = instructions.matrix_summary(trials)
    print("")
    print(f"matrix: {summary['episodes']} episode(s) = {len(summary['models'])} "
          f"model(s) x {summary['prompts']} prompt(s) x "
          f"{summary['repeats']} repeat(s)")
    print(f"log folder: {args.log_folder}")
    if args.group_models:
        for model, folder in instructions.model_folders(args.models).items():
            print(f"  {model:<30} {os.path.join(args.log_folder, folder)}")
    print("")

    started = time.time()
    new_rows = []
    failures = []
    considered = []
    attempted = 0
    skipped = 0
    total = len(trials)
    unreachable = False
    try:
        for number, trial in enumerate(trials, 1):
            if args.limit and attempted >= args.limit:
                print(f"--limit {args.limit} reached; {total - number + 1} "
                      f"episode(s) not attempted")
                break
            attempted += 1
            considered.append(trial)
            run = trial.artefact_stem()
            if args.resume and run in previous:
                skipped += 1
                print(f"[{number}/{total}] skip (measured): {trial.label}")
                continue
            print(f"[{number}/{total}] {trial.label}")
            print(f"      task: {trial.prompt}")
            print(f"      run: {run}")
            row = measure_one(args, trial)
            new_rows.append(row)
            # The tables are rewritten from the run records after every episode, so
            # an interrupted batch still matches what is on disk.
            gpu_logs.write_tables(args.log_folder)

            if row["exit_code"] == 0:
                print(f"      ok in {row['seconds']}s (stop reason: "
                      f"{row['stop_reason']})")
            else:
                failures.append(trial)
                print(f"      FAILED (code {row['exit_code']}) after "
                      f"{row['seconds']}s")
                if row["exit_code"] == 3:
                    # The agent is gone: every further episode would be a hole in
                    # the table, and its model time would be spent unmeasured.
                    unreachable = True
                    print("the collector cannot measure - stopping the batch; "
                          "restart the host agent and re-run with --resume")
                    break
                if args.fail_fast:
                    print("--fail-fast: stopping the batch")
                    break
            print("")
    except KeyboardInterrupt:
        print("")
        print("interrupted - the running episode was killed and its envelope "
              "closed")
        gpu_logs.write_tables(args.log_folder)
        gpu_logs.write_samples(args.log_folder)
        return 130

    paths = gpu_logs.write_tables(args.log_folder)
    sample_path, sample_rows, unreadable = gpu_logs.write_samples(args.log_folder)
    print(f"{len(new_rows) - len(failures)}/{len(new_rows)} episode(s) measured in "
          f"{time.time() - started:.1f}s"
          + (f" ({skipped} skipped by --resume)" if skipped else ""))
    print(f"table:    {paths['episodes']}")
    print(f"cells:    {paths['cells']}")
    print(f"samples:  {sample_path} ({sample_rows} row(s)"
          + (f", {unreadable} unreadable line(s)" if unreadable else "") + ")")
    tree = gpu_logs.describe(args.log_folder)
    print(f"tree:     {tree['root']} - {tree['episodes']} run(s), "
          f"{tree['series']} series, {tree['baselines']} baseline(s)")
    if considered:
        # The batch's own aggregate of the attempted episodes - the same tables
        # `run_instruction_catalogue.py` writes, so the folder is complete.
        json_path, csv_path = results.write(args, considered)
        print("")
        print(f"aggregate: {json_path}")
        print(f"results:   {csv_path}")
    if failures:
        print(f"{len(failures)} episode(s) failed: "
              f"{'; '.join(trial.label for trial in failures)}")
    print("rebuild the energy tables at any time: python -m "
          f"instruction_catalogue.gpu_logs \"{args.log_folder}\"")
    return 3 if unreachable else 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
