"""Command line of the catalogue batch: the matrix selectors and the preview.

Everything that decides *which* episodes exist is resolved here, before the
runner touches the server: ``--model all`` against ``models.yaml``, ``--tasks``
and ``--levels`` against the catalogue, and ``--preset``/``--objective-map``
against `instructions.TASK_OBJECTIVES`. The episode subprocess then receives
fully resolved flags, so there is exactly one place where a default can turn into
a concrete episode - the matrix printed by ``--dry-run`` is therefore what the
batch will really run, not a second interpretation of the flags.
"""

import argparse
import os

import yaml

from farm_agent import config as farm_config

from . import instructions

#: ``--model all`` reads this file (next to the package, not ``project_code``).
MODELS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models.yaml")

#: Artefacts default next to the baseline logs (``experiments`` is git-ignored).
DEFAULT_LOG_FOLDER = "../experiments/instruction_catalogue"


def read_models(path=None):
    """``models.yaml`` -> ``{backend: [model ids]}`` (``backend`` in YAML order)."""
    path = os.path.abspath(path or MODELS_FILE)
    if not os.path.isfile(path):
        raise instructions.CatalogueError(
            f"models file not found: {path}; create it, or pass --model <id>")
    with open(path, "r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    if not isinstance(document, dict):
        raise instructions.CatalogueError(
            f"{path}: expected `backend: [model ids]` blocks, got "
            f"{type(document).__name__}")
    return document


def models_for_backend(document, backend, path=None):
    """The ``<backend>:`` list of ``models.yaml``, validated to be non-empty."""
    models = document.get(backend) or []
    if isinstance(models, (str, int, float)):
        models = [models]
    if not isinstance(models, list):
        raise instructions.CatalogueError(
            f"{path or 'models.yaml'}: `{backend}:` must be a list of model ids")
    models = [str(model).strip() for model in models if str(model).strip()]
    if not models:
        raise instructions.CatalogueError(
            f"`{backend}:` lists no models in {path or 'models.yaml'}; "
            f"add some there or pass --model <id>")
    return models


EPILOG = """\
examples:
  python -m instruction_catalogue --model all --dry-run
  python -m instruction_catalogue --model all --repeats 3
  python -m instruction_catalogue --vllm --model qwen3:8b --preset normal `
         --tasks harvest_crop_task --levels indirect,implicit --repeats 3
  python -m instruction_catalogue --nvidia --model all --tasks all --levels all
  python -m instruction_catalogue --smoke

the catalogue file is input: --instructions-file points at another
`tasks = [...]` file with the same shape as Indirect_instructions.py
"""


def build_parser():
    parser = argparse.ArgumentParser(
        prog="instruction_catalogue",
        description=("Run the instruction catalogue: every task group x directness "
                     "level x prompt phrasing of Indirect_instructions.py as its own "
                     "episode, then aggregate the per-episode summaries."),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=EPILOG,
    )

    what = parser.add_argument_group("what to run")
    what.add_argument("--model", type=str, default=None,
                      help="model id, or `all` for the list under the backend's key "
                           "in models.yaml (required unless --smoke/--list-* is used)")
    what.add_argument("--models-file", type=str, default=MODELS_FILE,
                      help="the `--model all` registry (default: models.yaml next to "
                           "this package)")
    what.add_argument("--instructions-file", type=str,
                      default=instructions.DEFAULT_INSTRUCTIONS_FILE,
                      help="catalogue file with `tasks = [{group: {level: [prompt]}}]` "
                           "(default: Indirect_instructions.py in project_code)")
    what.add_argument("--tasks", type=str, default="all",
                      help="`all` or a comma list of task groups of the catalogue")
    what.add_argument("--levels", type=str, default="all",
                      help="`all` or a comma list of direct, indirect, implicit")
    what.add_argument("--seed", type=int, default=1,
                      help="world seed of repeat 1 (repeat r uses seed + r - 1)")
    what.add_argument("--repeats", type=int, default=1,
                      help="episodes per (model, task, level, prompt) cell")

    world = parser.add_argument_group("what each episode is scored on")
    world.add_argument("--preset", type=str, default="default",
                       choices=["default", "easy", "normal", "hard"],
                       help="difficulty preset (the world: money, seeds, tank, "
                            "weather, pests, rovers); `default` keeps the per-task "
                            "mapping (easy/normal/hard), anything else forces that "
                            "world for every task - the objective keeps its own "
                            "deadline either way")
    world.add_argument("--objective-map", type=str, default="",
                       help="task=objective overrides, e.g. `harvest_crop_task="
                            "profit_hard,plant_crop_water_task=harvest_normal` "
                            "(goal only; the preset stays)")

    run = parser.add_argument_group("how to run it")
    run.add_argument("--max-steps", type=int, default=60,
                     help="maximum number of LLM steps per episode")
    run.add_argument("--llm-timeout", type=float, default=300.0,
                     help="seconds to wait for one LLM reply per episode (default 300; "
                          "60 s was too short for a 27B turn over a crowded proxy, and a "
                          "turn that times out ends the episode as `llm_error`)")
    run.add_argument("--max-advance-ticks", type=int, default=120,
                     help="cap one `advance` call at this many ticks (120 = one day, "
                          "0 = no cap)")
    run.add_argument("--trial-timeout", type=float, default=0.0,
                     help="hard wall-clock cap per episode in seconds (0 = off; a "
                          "stuck backend then hangs the batch)")
    run.add_argument("--log-folder", type=str, default=DEFAULT_LOG_FOLDER,
                     help="root folder of the artefacts: one sub-folder per model "
                          "(its sanitised id) plus the aggregated .json/.csv")
    run.add_argument("--flat-logs", action="store_true",
                     help="write every model's artefacts directly into --log-folder "
                          "instead of one sub-folder per model")
    run.add_argument("--tag", type=str, default="",
                     help="extra suffix for the aggregated file names")
    run.add_argument("--url", type=str, default=None,
                     help="farm-sim base URL (default: server_urls.farm_sim_url of "
                          "config.yaml, else http://127.0.0.1:3111)")
    llm = run.add_mutually_exclusive_group()
    llm.add_argument("--vllm", action="store_true",
                     help="run against the OpenAI-compatible server in the `vllm:` "
                          "block of config.yaml")
    llm.add_argument("--nvidia", action="store_true",
                     help="run against the hosted NVIDIA NIM endpoint in the "
                          "`nvidia:` block of config.yaml")
    run.add_argument("--resume", action="store_true",
                     help="skip episodes whose _summary.json already exists")
    run.add_argument("--retry-failed", action="store_true",
                     help="with --resume (which it turns on): re-run the episodes whose "
                          "summary records `llm_error`/`check_api_fail` instead of "
                          "skipping them, after moving their old artefacts into "
                          "`_failed_<stop_reason>/` next to the episode")
    run.add_argument("--limit", type=int, default=0,
                     help="stop after this many episodes (0 = the whole matrix)")
    run.add_argument("--fail-fast", action="store_true",
                     help="stop the batch at the first episode that exits non-zero")
    run.add_argument("--dry-run", action="store_true",
                     help="print the expanded matrix and exit (no server, no LLM)")
    run.add_argument("--smoke", action="store_true",
                     help="one scripted --check-api episode per task group (no LLM), "
                          "to prove the plumbing before a real batch")
    run.add_argument("--list-models", action="store_true",
                     help="print the curated models of every backend and what the "
                          "endpoint currently serves, then exit")
    run.add_argument("--list-tasks", action="store_true",
                     help="print the task groups, their levels and their resolved "
                          "preset/objective, then exit")
    return parser


def parse_args(argv=None):
    """Parse the batch flags and normalise the three implicit selectors."""
    parser = build_parser()
    args = parser.parse_args(argv)
    args.backend = "vllm" if args.vllm else "nvidia" if args.nvidia else "ollama"
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.limit < 0:
        parser.error("--limit must be >= 0 (0 runs the whole matrix)")
    if args.max_advance_ticks < 0:
        parser.error("--max-advance-ticks must be >= 0 (0 turns the cap off)")
    if args.trial_timeout < 0:
        parser.error("--trial-timeout must be >= 0 (0 turns the cap off)")
    if args.llm_timeout <= 0:
        parser.error("--llm-timeout must be > 0 (seconds to wait for one LLM reply)")
    # `--retry-failed` is a refinement of `--resume`, never a second mode: the
    # episodes it re-runs are the ones `--resume` would skip.
    if args.retry_failed:
        args.resume = True
    if args.smoke and args.model:
        parser.error("--smoke runs one scripted episode per task group and takes no "
                     "--model; drop one of the two")
    if not args.smoke and not (args.list_models or args.list_tasks):
        # Same rule as the single-episode CLI: an OpenAI-compatible backend may
        # take its model id from config.yaml, `--model` still wins.
        if not args.model and args.backend != "ollama":
            args.model = farm_config.openai_model(args.backend)
        if not args.model:
            parser.error("--model is required (a model id, or `all`) unless --smoke, "
                         "--list-models or --list-tasks is used")
    args.objective_map = instructions.parse_objective_map(args.objective_map)
    # One sub-folder per model is the default layout; `--flat-logs` restores the
    # single flat folder. Everything downstream reads this one attribute.
    args.group_models = not args.flat_logs
    return args


def print_models(args):
    """``--list-models``: the curated list per backend, plus what is served.

    The served half is a convenience for keeping ``models.yaml`` in sync (an
    Ollama tag that was renamed is the usual cause of an episode that dies on its
    first turn), so an unreachable endpoint only prints a note here.
    """
    from . import runner   # late import: the previews must not need the HTTP layer

    try:
        document = read_models(args.models_file)
    except instructions.CatalogueError as exc:
        print(f"error: {exc}")
        return 2
    print(f"models file: {os.path.abspath(args.models_file)}")
    print("(`--model all` expands to the list of the active backend)")
    for backend, models in document.items():
        marker = "->" if backend == args.backend else "  "
        listed = models if isinstance(models, list) else [models]
        print(f"{marker} {backend} ({len(listed)})")
        for model in listed:
            print(f"      {model}")
    source, served = runner.served_models(args)
    print("")
    if served is None:
        print(f"served models: could not be read ({source})")
        return 0
    print(f"served models ({source}, {len(served)}):")
    for model in served:
        print(f"      {model}")
    curated = document.get(args.backend) or []
    curated = curated if isinstance(curated, list) else [curated]
    missing = [str(model) for model in curated if str(model) not in served]
    if missing:
        print(f"note: the {args.backend} list names models the endpoint does not "
              f"report: {', '.join(missing)}")
    return 0


def print_tasks(catalogue, args):
    """``--list-tasks``: the groups, their levels and the resolved goal/world."""
    print(f"catalogue: {os.path.abspath(args.instructions_file)}")
    for task, levels in catalogue.items():
        objective, preset = instructions.objective_for(task, args.objective_map)
        print(f"{task}")
        print(f"  preset {preset}   objective {objective}")
        for level, prompts in levels.items():
            print(f"  {level:<9} {len(prompts)} prompt(s)")
    print("")
    print(f"task groups: {', '.join(catalogue)}")
    return 0


def main(argv=None):
    """Resolve the matrix, then print it (``--dry-run``) or run it."""
    from . import runner   # imported late: --dry-run needs no HTTP layer

    farm_config.load_config()
    args = parse_args(argv)
    try:
        catalogue = instructions.load_catalogue(args.instructions_file)
        groups = instructions.resolve_task_groups(catalogue, args.tasks)
        levels = instructions.resolve_levels(args.levels)
        if args.list_models:
            return print_models(args)
        if args.list_tasks:
            return print_tasks(catalogue, args)
        if args.smoke:
            args.models = ["check_api"]
            trials = instructions.smoke_trials(catalogue, groups, args.preset,
                                               args.objective_map, args.seed)
        else:
            if str(args.model).strip().lower() == "all":
                args.models = models_for_backend(read_models(args.models_file),
                                                 args.backend, args.models_file)
            else:
                args.models = [args.model]
            trials = instructions.expand_trials(
                catalogue, args.models, args.backend, groups, levels, args.preset,
                args.objective_map, args.seed, args.repeats)
        if args.group_models:
            # Two model ids that sanitise alike would silently share a folder.
            instructions.model_folders(args.models)
    except instructions.CatalogueError as exc:
        print(f"error: {exc}")
        return 2

    if args.dry_run:
        return instructions.print_matrix(trials, args)
    return runner.run(args, trials)

