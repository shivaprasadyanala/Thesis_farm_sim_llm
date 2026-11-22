"""`main()` - argument handling, the fail-fast server check and the dispatch.

This is the body of the old script's `main`: everything here is orchestration,
the work lives in the modules it calls.
"""

import json
import logging
import time

from . import agent, api, check_api, config, episode, objectives, rendering, reporting, served, tools
from . import runtime

logger = logging.getLogger(__name__)


def main(argv=None):
    # `config.yaml` feeds `--url` (server_urls.farm_sim_url) and the Ollama host,
    # so it is read before anything else - and a broken file still exits here.
    config.load_config()
    args = config.parse_args(argv)
    runtime.base_url = config.resolve_base_url(args)
    runtime.MAX_ADVANCE_TICKS = args.max_advance_ticks

    print(f"farm-sim: {runtime.base_url}")
    print(f"model: {args.model or '(none - check-api)'} @ {config.llm_endpoint(args)}")

    # Fail fast: no log file or episode reset happens when the server is down.
    try:
        health = api.api_get("/api/v1/health")
    except Exception as exc:
        print(f"cannot reach the farm-sim server at {runtime.base_url}: {exc}")
        print('start it with: cd ../farm_sim && PORT=3111 npm start')
        return 1

    # The health payload is the ground truth for which world this run drives:
    # a server started with a different PORT/PRESET is a different game.
    print(api.describe_server(health))
    print(f"open {runtime.base_url} in the browser to watch this episode")
    if not health.get("clients"):
        print(f"note: no browser is connected to {runtime.base_url} yet")

    if args.list_objectives:
        objectives.print_objectives()
        return 0

    # Read the catalogue before anything is reset: the server replaces an unknown
    # objective id with `starter_easy`, so a typo would otherwise score something
    # else entirely. This also names the run's preset/objective mix up front.
    objectives.load_objective_catalogue()
    exit_code = objectives.validate_objective(args)
    if exit_code:
        return exit_code

    # The served crop economics and preset constants, so the prompt quotes the
    # numbers this server actually scores instead of stale literals.
    served.load_server_config()

    stem, log_path = config.setup_logging(args)
    print(f"log file: {log_path}")
    logger.info(f"farm-sim base url: {runtime.base_url}")
    logger.info(api.describe_server(health))
    logger.info(f"Running model: {args.model} (backend: {args.backend})")
    logger.info(f"args: {vars(args)}")
    # Recorded in the log, not just in the prompt: a server restarted with other
    # constants makes every number below stale, and this is where that shows.
    crops = runtime.server_config.get("crops") or {}
    print(f"served config: {len(crops)} crops, "
          f"{runtime.server_config.get('minutes_per_tick', '?')} min/tick, "
          f"water_per_action={served.litres_per_watering()}")
    logger.info(f"served config keys: {sorted(runtime.server_config)}")
    logger.info(f"served crops: {json.dumps(crops, sort_keys=True)}")
    print(objectives.describe_objective(args))
    logger.info(objectives.describe_objective(args))
    print(f"advance cap: {runtime.MAX_ADVANCE_TICKS or 'off'} tick(s) per call, deadline-aware")
    logger.info(f"advance cap: {runtime.MAX_ADVANCE_TICKS} tick(s) per call (0 = no cap)")
    # The same number the loop enforces, published in every observation so the model
    # can budget its steps instead of discovering the limit by being stopped.
    runtime.MAX_STEPS = args.max_steps
    print(f"step budget: {runtime.MAX_STEPS} LLM step(s), reported in every observation")
    logger.info(f"step budget: {runtime.MAX_STEPS} LLM step(s), reported in every observation")

    if args.check_api:
        started = time.time()
        exit_code = check_api.run_scripted_check(args)
        runtime.episode_seconds = time.time() - started
        reporting.summarise_run(args, stem)
        return exit_code

    episode.prepare_episode(args)
    schemas = tools.fetch_tool_schemas()
    print(f"tools from the server: {[entry['function']['name'] for entry in schemas]}")
    logger.info(f"tools from the server: {[entry['function']['name'] for entry in schemas]}")
    print(f"initial state:\n{rendering.state_to_text(runtime.state)}")
    logger.info(f"initial state:\n{rendering.state_to_text(runtime.state)}")

    started = time.time()
    # Ollama keeps its own loop; both OpenAI-compatible backends (`--vllm` and
    # `--nvidia`) share one, and `args.backend` tells it which config block to read.
    runner = agent.run_agent if args.backend == "ollama" else agent.run_agent_openai
    runner(args, schemas)
    # Recorded before the summary is written: `episode_time` is the agent loop
    # (every model turn plus every action request), not the bookkeeping around it -
    # the health check, the reset, the schema fetch and the plots are outside it.
    runtime.episode_seconds = time.time() - started
    reporting.summarise_run(args, stem)

    print(f"episode finished in {runtime.episode_seconds:.1f}s (agent loop)")
    logger.info(f"episode finished in {runtime.episode_seconds:.1f}s (agent loop)")

    if args.reset_after:
        episode.prepare_episode(args)
        print("episode reset after the run")
    return 0
