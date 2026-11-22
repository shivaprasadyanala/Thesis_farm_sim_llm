"""Pre-flight, one subprocess per episode, and the batch bookkeeping.

The episode is *always* a subprocess of the unchanged single-episode entry script
(``agentic_approach_baseline_farm_sim.py``) - never an in-process call. Three
reasons, all of them lessons from batching:

* the harness keeps one episode's state in module globals (`farm_agent.runtime`),
  so two episodes in one process would share counters, tokens and the visited
  tiles; a fresh interpreter per episode is the cheapest way to guarantee the
  isolation the summary promises;
* the child writes exactly the artefacts the single-episode docs describe
  (``.log``, ``_summary.json``, ``_latency.png``), one set per episode, because
  its own ``--tag`` makes the stem unique - no post-processing has to rename
  anything;
* a crashing or hanging episode (a backend that stalls, a server restart) costs
  one subprocess, not the batch.

The pre-flight deliberately repeats the single-episode checks instead of trusting
them: the matrix is expanded *before* any episode runs, so an objective id the
server does not know (which it silently replaces with ``starter_easy``) or a model
the endpoint does not serve is caught while the fix still costs nothing.
"""

import os
import subprocess
import sys
import time

import requests

from farm_agent import api, config as farm_config, objectives, runtime

from . import instructions, results

#: ``project_code``: where the entry script, ``config.yaml`` and ``farm_agent`` are,
#: and the working directory of every episode (so ``--log-folder`` relativises the
#: same way it does for a hand-started single episode).
PROJECT_CODE = instructions.BASE_DIR
EPISODE_SCRIPT = os.path.join(PROJECT_CODE, "agentic_approach_baseline_farm_sim.py")


def _ollama_models():
    base = farm_config.ollama_url()
    if not base:
        return "server_urls.ollama_url missing in config.yaml", None
    url = f"{str(base).rstrip('/')}/api/tags"
    try:
        payload = requests.get(url, timeout=10).json()
    except Exception as exc:
        return f"GET {url} failed: {exc}", None
    names = [entry.get("name") or entry.get("model")
             for entry in payload.get("models") or []]
    return url, [name for name in names if name]


def _openai_models(backend):
    base = farm_config.openai_base_url(backend)
    if not base:
        return f"{backend}.base_url missing in config.yaml", None
    url = f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"
    headers = {"Authorization": f"Bearer {farm_config.openai_api_key(backend)}"}
    try:
        payload = requests.get(url, headers=headers, timeout=10).json()
    except Exception as exc:
        return f"GET {url} failed: {exc}", None
    names = [entry.get("id") for entry in payload.get("data") or []]
    return url, [name for name in names if name]


def served_models(args):
    """``(source, model ids)`` the active backend reports, or ``(reason, None)``.

    Ollama answers ``/api/tags``, the two OpenAI-compatible endpoints
    ``/v1/models``. Only used for warnings: ``--model all`` is the curated list of
    ``models.yaml``, not "whatever the endpoint happens to serve" (a hosted NIM key
    can call dozens of models, and each one multiplies the matrix).
    """
    if getattr(args, "backend", "ollama") == "ollama":
        return _ollama_models()
    return _openai_models(args.backend)


def preflight(args, trials):
    """Check the server, the objectives and the models before episode 1.

    Exit codes match the single-episode CLI on purpose (1 = server unreachable,
    2 = an objective the server would silently replace), so a wrapper script sees
    the same numbers whether it started one episode or a hundred.
    """
    runtime.base_url = farm_config.resolve_base_url(args)
    print(f"farm-sim: {runtime.base_url}")
    endpoint = ("(scripted check, no LLM)" if args.smoke
                else f"@ {farm_config.llm_endpoint(args)}")
    print(f"LLM backend: {args.backend} {endpoint}")
    try:
        health = api.api_get("/api/v1/health")
    except Exception as exc:
        print(f"cannot reach the farm-sim server at {runtime.base_url}: {exc}")
        print("start it with: cd ../farm_sim; $env:PORT=3111; npm start")
        return 1
    print(api.describe_server(health))
    print(f"open {runtime.base_url} in the browser to watch the episodes")
    if not health.get("clients"):
        print(f"note: no browser is connected to {runtime.base_url} yet")

    objectives.load_objective_catalogue()
    ids = [entry["id"] for entry in runtime.objective_catalog]
    wanted = list(dict.fromkeys(trial.objective for trial in trials))
    if ids:
        unknown = [objective for objective in wanted if objective not in ids]
        if unknown:
            print(f"unknown objective(s): {', '.join(unknown)}")
            print(f"available: {', '.join(ids)}")
            print("the server replaces an unknown id with starter_easy, so the "
                  "batch stops here instead of scoring a different goal")
            return 2
        for objective in wanted:
            entry = next(item for item in runtime.objective_catalog
                         if item["id"] == objective)
            print(f"objective {objective}: deadline day "
                  f"{entry.get('deadline_day', '?')}, reward "
                  f"${entry.get('reward', '?')}")
    else:
        print("note: the server lists no objectives - skipping that validation")

    if args.smoke:
        return 0

    source, served = served_models(args)
    if served is None:
        print(f"note: cannot verify the models of the backend ({source})")
        return 0
    missing = [model for model in args.models if model not in served]
    if missing:
        print(f"warning: the backend does not report {', '.join(missing)} "
              f"(see --list-models); those episodes will probably fail")
    print(f"models: {len(args.models)} requested, "
          f"{len(args.models) - len(missing)} reported by the endpoint")
    return 0


def build_command(args, trial):
    """The argv of one episode: fully resolved flags, no inherited defaults.

    ``--preset`` and ``--objective`` are both passed even when they agree with the
    preset's default, because the batch claims a specific world/goal pair per
    episode and the child's own verification then checks that claim.
    """
    command = [sys.executable, EPISODE_SCRIPT,
               "--preset", trial.preset,
               "--objective", trial.objective,
               "--seed", str(trial.seed),
               "--task", trial.prompt,
               "--tag", trial.tag,
               "--log-folder", trial.folder(args.log_folder, args.group_models),
               "--max-steps", str(args.max_steps),
               "--llm-timeout", str(args.llm_timeout),
               "--max-advance-ticks", str(args.max_advance_ticks)]
    if args.smoke:
        command.append("--check-api")
    else:
        command += ["--model", trial.model]
    if args.backend == "vllm":
        command.append("--vllm")
    elif args.backend == "nvidia":
        command.append("--nvidia")
    if args.url:
        command += ["--url", args.url]
    return command


def run(args, trials):
    """Pre-flight, one subprocess per episode, then the aggregate.

    Returns 0 only when every attempted episode exited 0 *and* recorded an
    episode that actually ran to a stopping rule: the single-episode script
    returns 0 for an episode the backend answered with ``llm_error`` too (the
    summary it writes is data about the model), and a batch of 300 such episodes
    must not report success. The aggregate is written either way.
    """
    exit_code = preflight(args, trials)
    if exit_code:
        return exit_code

    # A single episode treats `--log-folder` as relative to its own working
    # directory (`project_code`); the batch may be started from anywhere, so the
    # folder is frozen here and passed down absolute.
    args.log_folder = os.path.abspath(args.log_folder)
    os.makedirs(args.log_folder, exist_ok=True)
    summary = instructions.matrix_summary(trials)
    print(f"matrix: {summary['episodes']} episode(s) = {len(summary['models'])} "
          f"model(s) x {summary['prompts']} prompt(s) x "
          f"{summary['repeats']} repeat(s)")
    print(f"log folder: {args.log_folder}")
    if args.group_models:
        for model, folder in instructions.model_folders(args.models).items():
            print(f"  {model:<30} {os.path.join(args.log_folder, folder)}")
    print("")

    started = time.time()
    failures = []
    considered = []
    attempted = 0
    total = len(trials)
    for number, trial in enumerate(trials, 1):
        if args.limit and attempted >= args.limit:
            print(f"--limit {args.limit} reached; {total - number + 1} episode(s) "
                  f"not attempted")
            break
        if args.resume and os.path.isfile(
                trial.summary_path(args.log_folder, args.group_models)):
            considered.append(trial)
            attempted += 1
            print(f"[{number}/{total}] skip (recorded): {trial.label}")
            continue
        considered.append(trial)
        attempted += 1
        print(f"[{number}/{total}] {trial.label}")
        print(f"      task: {trial.prompt}")
        episode_started = time.time()
        try:
            code = subprocess.run(build_command(args, trial), cwd=PROJECT_CODE,
                                  timeout=args.trial_timeout or None).returncode
        except subprocess.TimeoutExpired:
            code = -1
            print(f"      episode exceeded --trial-timeout {args.trial_timeout:.0f}s")
        except OSError as exc:
            code = -1
            print(f"      cannot start the episode: {exc}")
        elapsed = time.time() - episode_started

        stop_reason = None
        if code == 0:
            recorded = results.load_summary(
                trial.summary_path(args.log_folder, args.group_models))
            if recorded is None:
                # A zero exit without the artefact would make `--resume` re-run
                # the episode forever and hide it from the aggregate.
                code = -2
                print("      exited 0 but wrote no _summary.json - counting it "
                      "as a failure")
            else:
                stop_reason = recorded.get("stop_reason")
                if stop_reason in results.FAILED_STOP_REASONS:
                    code = -3
                    print(f"      exited 0 but the episode stopped with "
                          f"{stop_reason!r}")

        if code == 0:
            print(f"      ok in {elapsed:.1f}s (stop reason: {stop_reason})")
        else:
            failures.append(trial)
            print(f"      FAILED (code {code}) after {elapsed:.1f}s")
            if args.fail_fast:
                print("--fail-fast: stopping the batch")
                break
        print("")

    print(f"{attempted - len(failures)}/{attempted} episode(s) ok in "
          f"{time.time() - started:.1f}s")
    json_path, csv_path = results.write(args, considered)
    print("")
    print(f"aggregate: {json_path}")
    print(f"table:     {csv_path}")
    if failures:
        print(f"{len(failures)} episode(s) failed: "
              f"{'; '.join(trial.label for trial in failures)}")
        return 1
    return 0