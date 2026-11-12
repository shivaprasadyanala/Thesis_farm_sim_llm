"""`config.yaml`, the command line and the log file."""

import argparse
import logging
import os
import sys

import yaml

from . import runtime

#: Folder that holds this harness (``.../project_code``): `config.yaml` is read
#: from here, while `--log-folder` stays relative to the working directory - the
#: same two anchors the single-file script used.
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

logger = logging.getLogger(__name__)

#: Default task text handed to the model. `None` means "use the objective this
#: preset assigns" (`game/phaser_test/farm_sim/sim/systems/objective.js`), so the
#: prompt always asks for exactly what the server scores:
#:   easy   -> starter_easy    "Plant 2 crops and keep them watered."
#:   normal -> harvest_normal  "Harvest 3 crops before the end of day 10."
#:   hard   -> profit_hard     "Earn $150 from sales and harvest 4 crops before day 18."
#: `--objective <id>` replaces the goal while keeping the preset's world, and
#: `--task` only rewrites the instruction text (the objective stays authoritative).
DEFAULT_TASK = None


def read_config(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)  # safe_load prevents code execution
            return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error reading file: {e}")
        sys.exit(1)


def load_config(file_path=None):
    """Read `config.yaml` into `runtime.config_data`; called once from `main`."""
    path = file_path or os.path.join(BASE_DIR, "config.yaml")
    runtime.config_data = read_config(path) or {}
    return runtime.config_data


def ollama_url():
    """`server_urls.ollama_url` of config.yaml (None when the key is missing)."""
    return (runtime.config_data.get("server_urls") or {}).get("ollama_url")


def vllm_config():
    """`vllm:` block of config.yaml - the OpenAI-compatible server of `--vllm`."""
    return runtime.config_data.get("vllm") or {}


def vllm_base_url():
    """`vllm.base_url` without a trailing slash ('' when the key is missing)."""
    return str(vllm_config().get("base_url") or "").rstrip("/")


def vllm_api_key():
    """`vllm.api_key`; the OpenAI SDK wants *some* key, even a placeholder."""
    return vllm_config().get("api_key") or "EMPTY"


def vllm_model():
    """`vllm.model` - the served model id, used when `--vllm` comes without --model."""
    return vllm_config().get("model")


def vllm_enable_thinking():
    """`vllm.enable_thinking`: None means "send no chat_template_kwargs at all"."""
    return vllm_config().get("enable_thinking")


def llm_endpoint(args):
    """Where the run sends its prompts: the vLLM proxy (`--vllm`) or Ollama."""
    if args.vllm:
        return vllm_base_url() or "(vllm.base_url missing in config.yaml)"
    return ollama_url()


def parse_args(argv=None):

    parser = argparse.ArgumentParser(
        description="Agentic baseline for the farm simulation (farm-sim REST API).",
    )
    parser.add_argument("--model", type=str, default=None,
                        help="LLM model name served by Ollama (required unless --check-api)")
    parser.add_argument("--url", type=str, default=None,
                        help="farm-sim base URL (default: config server_urls.farm_sim_url "
                             "or http://127.0.0.1:3111)")
    parser.add_argument("--preset", type=str, default="easy",
                        choices=["easy", "normal", "hard"], help="difficulty preset of the episode")
    parser.add_argument("--objective", type=str, default=None,
                        help="objective id from the server (see --list-objectives), e.g. "
                             "starter_easy, harvest_normal, profit_hard; it replaces the goal "
                             "only - the preset still chooses the world "
                             "(default: the objective the preset assigns)")
    parser.add_argument("--list-objectives", action="store_true",
                        help="print the objective catalogue of the server and exit")
    parser.add_argument("--seed", type=int, default=1, help="world seed of the episode")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK,
                        help="task text given to the model and stored in the sim "
                             "(default: the objective the preset assigns)")
    # 60, not 50: the recorded `hard` episode's critical path was 54 steps (plant
    # day 1 -> two corn harvests -> carry -> two `sell`s -> the second wheat
    # harvest -> its sale) and it ended at step 50 holding $167 of unsold produce,
    # four steps short. `normal` finishes in 30-45, so the extra headroom is only
    # spent on the preset that needs it.
    parser.add_argument("--max-steps", type=int, default=60, help="maximum number of LLM steps")
    parser.add_argument("--llm-timeout", type=float, default=60.0,
                        help="seconds to wait for one LLM reply (default 60). Raise it when the "
                             "Ollama host has fallen back to CPU: a 26B model then needs longer "
                             "than a minute just to evaluate the prompt, and the run dies as "
                             "`llm_error` with `size_vram: 0` in /api/ps")
    parser.add_argument("--log-folder", type=str,
                        default="../experiments/test_logs_baseline_farm_sim",
                        help="folder for the .log/.png/.json artefacts")
    parser.add_argument("--tag", type=str, default="", help="extra suffix for the artefact names")
    parser.add_argument("--no-reset", action="store_true",
                        help="keep the episode that is already running on the server")
    parser.add_argument("--reset-after", action="store_true",
                        help="start another fresh episode after the run (leaves the game clean)")
    parser.add_argument("--max-advance-ticks", type=int, default=120,
                        help="cap one `advance` call at this many ticks (120 = one in-game "
                             "day, 0 = no cap); the harness also never advances past the "
                             "objective deadline")
    parser.add_argument("--check-api", action="store_true",
                        help="skip the LLM and drive the API with a fixed script (smoke test)")
    parser.add_argument("--vllm", action="store_true",
                        help="send the prompts to the OpenAI-compatible server in the "
                             "`vllm:` block of config.yaml instead of Ollama (model "
                             "defaults to `vllm.model` when --model is omitted)")
    args = parser.parse_args(argv)
    # `--vllm` may take its model id from config.yaml; `--model` still wins, so an
    # explicit Ollama model is never silently swapped for the served one.
    if args.vllm and not args.model:
        args.model = vllm_model()
    if not args.check_api and not args.list_objectives and not args.model:
        parser.error("--model is required unless --check-api or --list-objectives is used")

    if args.max_advance_ticks < 0:
        parser.error("--max-advance-ticks must be >= 0 (0 turns the cap off)")
    return args


def resolve_base_url(args):
    if args.url:
        return args.url.rstrip("/")
    configured = (runtime.config_data.get("server_urls") or {}).get("farm_sim_url")
    if configured:
        return str(configured).rstrip("/")
    return "http://127.0.0.1:3111"


def setup_logging(args):
    """Same log layout as the legacy baseline: one file per model/preset/seed.

    `filemode='w'` on purpose: the default `'a'` appended every run into the same
    `model_preset_seed.log`, so the recorded logs mixed up to 20 episodes (and
    their objectives) in one file - impossible to attribute after the fact. The
    artefacts of a run are now one `.log`, one `_summary.json` and one `.png`.
    """
    os.makedirs(args.log_folder, exist_ok=True)
    stem = f"{args.model or 'check_api'}_{args.preset}_seed{args.seed}{args.tag}"
    stem = stem.replace(":", "_").replace(".", "_").replace("/", "_").replace("\\", "_")
    log_path = os.path.join(args.log_folder, f"{stem}.log")
    logging.basicConfig(filename=log_path, encoding='utf-8', level=logging.INFO,
                        filemode='w',
                        format="%(asctime)s - %(levelname)s - %(message)s")
    logging.getLogger("httpx").disabled = True
    logging.getLogger("httpcore").disabled = True
    return stem, log_path
