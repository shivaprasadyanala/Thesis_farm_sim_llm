"""The instruction catalogue: what to ask, how directly, and which goal it maps to.

``../Indirect_instructions.py`` - the data file this repository already had - is
the input of the experiment::

    tasks = [
        {"plant_crop_water_task": {
            "direct":   ["Plant two crops and keep both watered.", ...],
            "indirect": [...],
            "implicit": [...],
        }},
        ...
    ]

It stays a plain data file and is *imported*, never copied or rewritten: it is
read once per batch, validated, and flattened into the list of `Trial` objects
the runner executes. Nothing in this module touches the network or the harness,
which is why the whole matrix can be printed (``--dry-run``) without a server.

Three decisions live here and nowhere else:

1. **Task -> objective/preset.** The catalogue's three task groups are the three
   objectives of ``sim/systems/objective.js``, and their deadlines are exactly
   the days the prompts talk about (day 8 / day 10 / day 18), so the default
   mapping is ``plant_crop_water_task -> starter_easy`` (easy),
   ``harvest_crop_task -> harvest_normal`` (normal) and
   ``harvest_crop_earn_money_task -> profit_hard`` (hard). ``--objective-map``
   rewrites the goal id only, ``--preset`` rewrites the world only: the two are
   independent because ``POST /api/v1/reset`` takes them independently, and each
   objective carries its own ``deadline_day``/``reward``.
2. **Level order.** ``direct, indirect, implicit`` - the catalogue's own order,
   so ``--levels all`` reads the way the file does.
3. **Trial order.** model -> task -> level -> prompt -> repeat. A model's whole
   cell finishes before the next model starts, so a batch interrupted half-way
   (or cut with ``--limit``) is still a balanced prefix of the matrix, and
   ``--resume`` can pick up exactly where it stopped.

Repeat ``r`` of a cell uses seed ``--seed + r - 1`` and *all* phrasings of that
cell share the same repeat seeds. Direct/indirect/implicit are therefore compared
on identical worlds - a paired design - instead of each wording also getting a
different farm.
"""

import importlib.util
import os
from collections import OrderedDict
from dataclasses import dataclass

#: ``project_code`` - the folder the entry script and ``config.yaml`` live in.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: The directness levels, in catalogue order. ``--levels all`` expands to this.
LEVELS = ("direct", "indirect", "implicit")

#: Default catalogue file: the one that shipped with the repository.
DEFAULT_INSTRUCTIONS_FILE = os.path.join(BASE_DIR, "Indirect_instructions.py")

#: Task group -> (objective id, preset) used when the flag leaves it at default.
#: Kept as an OrderedDict because ``--tasks all`` and the printed matrix follow
#: this order.
TASK_OBJECTIVES = OrderedDict([
    ("plant_crop_water_task", ("starter_easy", "easy")),
    ("harvest_crop_task", ("harvest_normal", "normal")),
    ("harvest_crop_earn_money_task", ("profit_hard", "hard")),
])


class CatalogueError(Exception):
    """Anything wrong with the catalogue file or with a task/level selection."""


def sanitise(stem):
    """The character set ``farm_agent.config.setup_logging`` uses for artefacts.

    Model ids carry ``:`` (``qwen3:8b``), ``.`` (``llama3.3:latest``) and ``/``
    (``Qwen/Qwen3.8-27B``); Windows file names allow none of them, so the batch
    layer has to name files exactly the way the child process will, or
    ``--resume`` and the aggregation would look for artefacts that never appear.
    """
    return (stem.replace(":", "_").replace(".", "_")
            .replace("/", "_").replace("\\", "_"))


def model_folder(model):
    """The sub-folder a model's episodes are written to (its sanitised id)."""
    return sanitise(model or "check_api")


def model_folders(models):
    """``{model: folder}`` for the batch, refusing two models that collide.

    ``Qwen/Qwen3.8-27B`` and ``Qwen_Qwen3_8-27B`` both sanitise to the same folder
    name; silently mixing their artefacts would corrupt both the resume check and
    the aggregate, so a collision stops the batch before episode 1.
    """
    folders = OrderedDict((model, model_folder(model)) for model in models)
    seen = {}
    for model, folder in folders.items():
        if folder in seen:
            raise CatalogueError(
                f"models {seen[folder]!r} and {model!r} map to the same log "
                f"folder {folder!r}; rename one or pass --flat-logs")
        seen[folder] = model
    return folders


def load_catalogue(path=None):
    """Read ``tasks`` from the catalogue file; ``OrderedDict[task][level] = [prompts]``.

    Loaded by path with ``importlib`` rather than as a package module: the file is
    a data file someone edits between experiments (or drops in from a paper
    appendix), not part of the package, and ``--instructions-file`` has to be able
    to point anywhere.

    Every problem is a `CatalogueError` naming the offending entry, because the
    failure mode this guards against is silent: a level that vanishes from a task
    group would otherwise simply produce fewer episodes than the matrix claims.
    """
    path = os.path.abspath(path or DEFAULT_INSTRUCTIONS_FILE)
    if not os.path.isfile(path):
        raise CatalogueError(f"instruction catalogue not found: {path}")
    spec = importlib.util.spec_from_file_location("instruction_catalogue_data", path)
    if spec is None or spec.loader is None:
        raise CatalogueError(f"cannot load {path} as a Python file")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise CatalogueError(f"cannot import {path}: {exc}")

    entries = getattr(module, "tasks", None)
    if not isinstance(entries, list) or not entries:
        raise CatalogueError(f"{path} defines no non-empty `tasks` list")

    catalogue = OrderedDict()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or len(entry) != 1:
            raise CatalogueError(
                f"{path}: entry {index + 1} must be a dict with exactly one task "
                f"group as its key, got {type(entry).__name__} "
                f"with {len(entry) if isinstance(entry, dict) else '?'} key(s)")
        task, levels = next(iter(entry.items()))
        if task in catalogue:
            raise CatalogueError(f"{path}: task group {task!r} appears more than once")
        if not isinstance(levels, dict) or not levels:
            raise CatalogueError(f"{path}: task group {task!r} has no levels")
        unknown = [level for level in levels if level not in LEVELS]
        if unknown:
            raise CatalogueError(
                f"{path}: task group {task!r} has unknown level(s) {unknown}; "
                f"known levels: {', '.join(LEVELS)}")
        cleaned = OrderedDict()
        for level in LEVELS:
            if level not in levels:
                continue
            prompts = levels[level]
            if isinstance(prompts, str):
                prompts = [prompts]
            if not isinstance(prompts, list) or not prompts:
                raise CatalogueError(
                    f"{path}: {task!r}/{level!r} must be a non-empty list of prompts")
            for number, prompt in enumerate(prompts, 1):
                if not isinstance(prompt, str) or not prompt.strip():
                    raise CatalogueError(
                        f"{path}: {task!r}/{level!r} prompt {number} is not a "
                        f"non-empty string")
            cleaned[level] = [prompt.strip() for prompt in prompts]
        catalogue[task] = cleaned
    return catalogue


def _split(spec):
    """``"a, B ,a"`` -> ``["a", "B"]`` (order kept, blanks and duplicates dropped)."""
    wanted = []
    for part in str(spec or "").replace(";", ",").split(","):
        part = part.strip()
        if part and part not in wanted:
            wanted.append(part)
    return wanted


def _is_all(spec):
    return str(spec or "").strip().lower() in ("", "all", "*")


def resolve_task_groups(catalogue, spec):
    """Expand ``--tasks`` (``all`` or a comma list) against the catalogue."""
    if _is_all(spec):
        return list(catalogue)
    wanted, unknown = [], []
    lowered = {task.lower(): task for task in catalogue}
    for part in _split(spec):
        task = lowered.get(part.lower())
        if task is None:
            unknown.append(part)
        elif task not in wanted:
            wanted.append(task)
    if unknown:
        raise CatalogueError(
            f"unknown task group(s) {unknown}; the catalogue has: "
            f"{', '.join(catalogue)}")
    return wanted


def resolve_levels(spec):
    """Expand ``--levels`` (``all`` or a comma list) to the catalogue's order."""
    if _is_all(spec):
        return list(LEVELS)
    wanted, unknown = [], []
    for part in _split(spec):
        lowered = part.lower()
        if lowered not in LEVELS:
            unknown.append(part)
        elif lowered not in wanted:
            wanted.append(lowered)
    if unknown:
        raise CatalogueError(
            f"unknown level(s) {unknown}; known levels: {', '.join(LEVELS)}")
    if not wanted:
        raise CatalogueError("--levels selected no level")
    return [level for level in LEVELS if level in wanted]


def parse_objective_map(spec):
    """``--objective-map harvest_crop_task=profit_hard,...`` -> ``{task: objective}``.

    Only the goal id is rewritten. The preset (the world: money, seeds, tank,
    weather, pests, rovers) stays whatever the task's default mapping says, or
    whatever ``--preset`` forces - so "same world, different goal" and "same
    goal, different world" are both expressible without a new flag.
    """
    mapping = {}
    for part in _split(spec):
        if "=" not in part:
            raise CatalogueError(
                f"--objective-map expects task=objective pairs, got {part!r}")
        task, objective = (piece.strip() for piece in part.split("=", 1))
        if not task or not objective:
            raise CatalogueError(
                f"--objective-map expects task=objective pairs, got {part!r}")
        mapping[task] = objective
    return mapping


def objective_for(task, objective_map=None):
    """``(objective id, preset)`` a task group is scored on."""
    if objective_map and task in objective_map:
        return objective_map[task], TASK_OBJECTIVES.get(task, (None, None))[1]
    if task in TASK_OBJECTIVES:
        return TASK_OBJECTIVES[task]
    raise CatalogueError(
        f"task group {task!r} has no objective mapping; add it to "
        f"TASK_OBJECTIVES or pass --objective-map {task}=<objective id>")


@dataclass(frozen=True)
class Trial:
    """One episode: a model, a task, one wording, one repeat, and its world."""

    model: str          # model id, or "check_api" for --smoke
    backend: str        # ollama | vllm | nvidia
    task: str           # task group of the catalogue
    level: str          # direct | indirect | implicit
    prompt_index: int   # 1-based position inside the level's prompt list
    prompt: str         # the instruction text, passed verbatim to --task
    repeat: int         # 1-based repeat; selects the seed
    seed: int           # world seed handed to the episode
    preset: str         # difficulty preset (the world)
    objective: str      # objective id (the goal the server scores)

    @property
    def tag(self):
        """``--tag`` of the episode: the suffix that makes its artefacts unique."""
        return (f"_{self.task}_{self.level}"
                f"_p{self.prompt_index}_r{self.repeat}")

    @property
    def label(self):
        return (f"{self.model} | {self.task} | {self.level} | "
                f"p{self.prompt_index} | r{self.repeat} | seed {self.seed}")

    def artefact_stem(self):
        """The stem `farm_agent.config.setup_logging` will use for this episode.

        Mirrors ``f"{model}_{preset}_seed{seed}{tag}"`` plus the same character
        sanitising, because ``--resume`` and the aggregation look artefacts up by
        this name instead of asking the child process what it wrote.
        """
        return sanitise(f"{self.model or 'check_api'}_{self.preset}"
                        f"_seed{self.seed}{self.tag}")

    def folder(self, log_folder, group_models=True):
        """This episode's log folder: one sub-folder per model by default.

        ``group_models`` is what ``--flat-logs`` turns off (all models into one
        folder). The child creates the folder itself, so nothing here has to.
        """
        if group_models:
            return os.path.join(log_folder, model_folder(self.model))
        return log_folder

    def summary_path(self, log_folder, group_models=True):
        """Where the child's ``_summary.json`` for this trial will be."""
        return os.path.join(self.folder(log_folder, group_models),
                            f"{self.artefact_stem()}_summary.json")


def expand_trials(catalogue, models, backend, groups, levels, preset,
                  objective_map=None, seed=1, repeats=1):
    """Flatten the matrix into the ordered list of `Trial` objects to run.

    ``preset`` is either ``"default"`` (each task keeps its mapped preset) or a
    concrete name (one world for every task). ``repeats`` repeats a *cell*
    (model x task x level x prompt) with consecutive seeds, so the three levels
    of one prompt are always compared on the same three farms.
    """
    objective_map = objective_map or {}
    repeats = max(1, int(repeats))
    forced_preset = None if str(preset).lower() in ("default", "mapped", "") else preset
    trials = []

    # Fail before the first episode, not when the matrix silently gets shorter.
    for task in groups:
        missing = [level for level in levels if level not in catalogue[task]]
        if missing:
            raise CatalogueError(
                f"{task!r} has no {', '.join(missing)} prompt(s); "
                f"it has: {', '.join(catalogue[task])}")

    for model in models:
        for task in groups:
            objective, task_preset = objective_for(task, objective_map)
            world = forced_preset or task_preset
            if not world:
                raise CatalogueError(
                    f"{task!r} has no preset mapping; pass --preset easy|normal|hard")
            for level in levels:
                for index, prompt in enumerate(catalogue[task][level], 1):
                    for repeat in range(1, repeats + 1):
                        trials.append(Trial(
                            model=model,
                            backend=backend,
                            task=task,
                            level=level,
                            prompt_index=index,
                            prompt=prompt,
                            repeat=repeat,
                            seed=seed + repeat - 1,
                            preset=world,
                            objective=objective,
                        ))
    return trials


def smoke_trials(catalogue, groups, preset, objective_map=None, seed=1):
    """One scripted (``--check-api``) episode per task group, no LLM involved.

    ``--smoke`` exists to prove the plumbing - reset with ``--objective`` and
    ``--task``, the objective verification, the artefact naming, the collection -
    before a real batch spends hours of model time on it.
    """
    objective_map = objective_map or {}
    forced_preset = None if str(preset).lower() in ("default", "mapped", "") else preset
    trials = []
    for task in groups:
        objective, task_preset = objective_for(task, objective_map)
        level = next((name for name in LEVELS if name in catalogue[task]), LEVELS[0])
        trials.append(Trial(
            model="check_api",
            backend="check_api",
            task=task,
            level=level,
            prompt_index=1,
            prompt=catalogue[task][level][0],
            repeat=1,
            seed=seed,
            preset=forced_preset or task_preset,
            objective=objective,
        ))
    return trials


def matrix_summary(trials):
    """Counts for ``--dry-run`` and the batch header."""
    models = list(OrderedDict.fromkeys(trial.model for trial in trials))
    tasks = list(OrderedDict.fromkeys(trial.task for trial in trials))
    levels = list(OrderedDict.fromkeys(trial.level for trial in trials))
    prompts = len({(trial.task, trial.level, trial.prompt_index) for trial in trials})
    repeats = len({trial.repeat for trial in trials}) or 1
    return {
        "episodes": len(trials),
        "models": models,
        "tasks": tasks,
        "levels": levels,
        "prompts": prompts,
        "repeats": repeats,
    }


def print_matrix(trials, args):
    """``--dry-run``: the whole matrix, one line per episode, no server involved.

    Prints the resolved preset/objective of every task group and the verbatim
    prompt of every ``(task, level, prompt)`` cell, so a mistyped task name, a
    level that vanished from the catalogue, or a prompt that is not the one in
    the file all show up before any model time is spent.
    """
    summary = matrix_summary(trials)
    mapped_preset = str(args.preset).lower() in ("default", "mapped", "")
    print(f"catalogue: {os.path.abspath(args.instructions_file)}")
    print(f"backend: {args.backend}")
    print(f"models ({len(summary['models'])}): {', '.join(summary['models'])}")
    print(f"tasks ({len(summary['tasks'])}): {', '.join(summary['tasks'])}")
    print(f"levels ({len(summary['levels'])}): {', '.join(summary['levels'])}")
    print(f"preset: {'mapped per task' if mapped_preset else args.preset}")
    for task in summary["tasks"]:
        objective, preset = objective_for(task, args.objective_map)
        world = preset if mapped_preset else args.preset
        print(f"  {task:<28} {world:<7} {objective}")
    if args.objective_map:
        print(f"objective map: {args.objective_map}")
    print(f"seed: {args.seed} (+1 per repeat)   repeats: {summary['repeats']}")
    print(f"log folder: {os.path.abspath(args.log_folder)}")
    if getattr(args, "group_models", True):
        for model, folder in model_folders(summary["models"]).items():
            print(f"  {model:<30} "
                  f"{os.path.join(os.path.abspath(args.log_folder), folder)}")
    print("")
    print(f"{'#':>4}  {'model':<24} {'task':<28} {'level':<9} {'p':>2} {'r':>2} "
          f"{'seed':>4}  preset/objective")
    shown = None
    for number, trial in enumerate(trials, 1):
        print(f"{number:>4}  {trial.model:<24} {trial.task:<28} {trial.level:<9} "
              f"{trial.prompt_index:>2} {trial.repeat:>2} {trial.seed:>4}  "
              f"{trial.preset}/{trial.objective}")
        cell = (trial.task, trial.level, trial.prompt_index)
        if cell != shown:
            print(f'      "{trial.prompt}"')
        shown = cell
    print("")
    print(f"{summary['episodes']} episode(s) = {len(summary['models'])} model(s) x "
          f"{summary['prompts']} prompt(s) x {summary['repeats']} repeat(s)")
    return 0
