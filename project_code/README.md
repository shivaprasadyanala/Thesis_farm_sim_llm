# farm_agent — agentic baseline for farm-sim

An LLM agent that plays the server-authoritative farm simulation in `../farm_sim`.
One entry script, one package, one system prompt: the model gets the 18 farm
actions as tools, spends each turn on one or more of them, and every reply it sees
is the server's own observation - nothing is simulated in Python.

```
                    system prompt + tool catalogue
   ┌──────────┐  ───────────────────────────────────►  ┌──────────────┐
   │   LLM    │                                        │ farm_agent/  │
   │ Ollama / │  ◄─── the turn's tool calls ─────────  │   (harness)  │
   │ vLLM/NIM │                                        └──────┬───────┘
   └──────────┘                                               │ POST /api/v1/action
                                                              │ GET  /api/v1/state
                                                       ┌──────▼───────┐
                                                       │   farm_sim   │ 500 ms tick
                                                       │ server (JS)  │ owns every rule
                                                       └──────┬───────┘
                                                              │ SSE
                                                       ┌──────▼───────┐
                                                       │    Phaser    │ you can watch
                                                       └──────────────┘
```

The harness never posts its own world back: every tool call is one
`POST /api/v1/action`, the reply carries the fresh observation with it, and the
model reads a compact digest of that state on its next turn.

## What is where

| path | what it is |
| --- | --- |
| `agentic_approach_baseline_farm_sim.py` | the thin entry point (docstring = the long-form notes); `python -m farm_agent` runs the same `main` |
| `farm_agent/` | the harness, one module per concern (see [Module map](#module-map)) |
| `config.yaml` | server URLs, the `vllm:` and `nvidia:` blocks, legacy keys |
| `../experiments/test_logs_baseline_farm_sim/` | `.log`, `_latency.png` (per action), `_llm_latency.png` (per model turn), `_summary.json` of every run (`--log-folder`) |

## Quick start

```powershell
# terminal 1 - the simulation (a bare `npm start` serves 3100, the harness uses 3111)
cd ..\farm_sim
$env:PORT=3111; npm start              # http://127.0.0.1:3111
```

```bash
cd farm_sim && PORT=3111 npm start     # same thing in bash
```

```powershell
# terminal 2 - the agent
cd project_code
python agentic_approach_baseline_farm_sim.py --model qwen3:8b --preset easy --seed 1

# no LLM needed: drives till -> plant -> water -> advance -> harvest -> sell
python agentic_approach_baseline_farm_sim.py --check-api
```

Dependencies (whatever `pip install` into the interpreter you use - the repo has
a conda env in `my_env/`):

| package | version used here | for |
| --- | --- | --- |
| Python | 3.9.6 | - |
| `ollama` | 0.6.2 | the Ollama loop (`agent.py`) |
| `openai` | 2.24.0 | the `--vllm` / `--nvidia` loop (`agent.py`) |
| `requests` | 2.32.5 | the farm-sim HTTP layer (`api.py`) |
| `PyYAML` | 6.0.3 | `config.yaml` (`config.py`) |
| `matplotlib` / `numpy` | 3.9.4 / 2.0.2 | the latency plot and its statistics (`reporting.py`) |

## Commands

| flag | default | meaning |
| --- | --- | --- |
| `--model` | - | model name served by the backend; **required** unless `--check-api` / `--list-objectives`. With `--vllm` / `--nvidia` it defaults to `vllm.model` / `nvidia.model` |
| `--vllm` | off | talk to the OpenAI-compatible server in the `vllm:` block instead of Ollama |
| `--nvidia` | off | talk to the hosted NVIDIA NIM endpoint in the `nvidia:` block instead (same loop as `--vllm`, mutually exclusive with it) |
| `--url` | config / `:3111` | farm-sim base URL (`--url` > `server_urls.farm_sim_url` > `http://127.0.0.1:3111`) |
| `--preset` | `easy` | `easy`, `normal`, `hard` - chooses the **world** (money, seeds, tank, weather, pests, rovers, fog) |
| `--objective` | preset default | objective id from the server - chooses the **goal** only (`--list-objectives`) |
| `--list-objectives` | - | print the server's objective catalogue and each preset's default, then exit |
| `--seed` | `1` | RNG seed of the episode (weather, market, plot draw); `(seed, actions)` fully determines a run |
| `--task` | the objective text | rewrites the instruction wording; the objective stays what is scored |
| `--max-steps` | `60` | LLM steps; the remaining budget is printed in every observation |
| `--llm-timeout` | `300` | seconds per LLM reply; 60 s was too short for a 27B turn over a crowded proxy (a turn that times out ends the episode as `llm_error`). A transport failure is retried twice (5 s, 10 s) before the turn is given up, and `ollama.keep_alive` keeps the weights loaded between turns |
| `--max-advance-ticks` | `120` | cap on one `advance` (120 ticks = 1 game day, `0` = off); also capped at the objective deadline |
| `--log-folder` | `../experiments/test_logs_baseline_farm_sim` | where the artefacts go |
| `--tag` | empty | suffix for the artefact names (see [Outputs](#outputs)) |
| `--no-reset` | off | adopt the episode already running instead of starting a new one |
| `--reset-after` | off | start a fresh episode when the run ends (leaves the game clean) |
| `--check-api` | off | scripted smoke test: no LLM, same tools, same summary |

Examples:

```powershell
# the three presets with their default objectives
python agentic_approach_baseline_farm_sim.py --model qwen3:8b --preset easy   --seed 1
python agentic_approach_baseline_farm_sim.py --model qwen3:8b --preset normal --seed 2

# hold the goal fixed, vary the difficulty: the easy farm scored on the harvest goal
python agentic_approach_baseline_farm_sim.py --model qwen3:8b --preset easy \
    --objective harvest_normal --seed 3 --tag _harvest_normal

# what can the server score, and which goal does each preset pick?
python agentic_approach_baseline_farm_sim.py --list-objectives

# OpenAI-compatible vLLM behind the DGX proxy (model comes from config.yaml)
python agentic_approach_baseline_farm_sim.py --vllm --preset normal --seed 2 --llm-timeout 180
python agentic_approach_baseline_farm_sim.py --vllm --model Qwen/Qwen3.8-27B-FP8 --max-steps 30

# the same loop against NVIDIA NIM (model comes from config.yaml, --model wins)
python agentic_approach_baseline_farm_sim.py --nvidia --preset easy --seed 1 --llm-timeout 180
python agentic_approach_baseline_farm_sim.py --nvidia --model openai/gpt-oss-20b --max-steps 10

# continue the episode that is already running, then leave the game clean
python agentic_approach_baseline_farm_sim.py --model qwen3:8b --no-reset --reset-after

# smoke tests that need no model
python agentic_approach_baseline_farm_sim.py --check-api --preset normal
python -m farm_agent --check-api
```

## `config.yaml`

Only four blocks are read by this harness (everything else in the file belongs to
the older scripts); `vllm:` and `nvidia:` have the same shape, because both are
OpenAI-compatible servers:

| key | used for |
| --- | --- |
| `server_urls.farm_sim_url` | default base URL of the simulation (`http://127.0.0.1:3111`) |
| `server_urls.ollama_url` | Ollama host for the default loop (`http://hal9000.skim.th-owl.de:11430`) |
| `ollama.keep_alive` | `/api/chat` field: how long the weights stay loaded (`"24h"`, `-1` = forever). Without it Ollama unloads after five idle minutes and the reload is charged to the next turn's `--llm-timeout` |
| `ollama.think` | `false` asks the template to skip the reasoning trace (the `/api/chat` twin of `vllm.enable_thinking`); `null` sends nothing |
| `ollama.num_ctx` | KV-cache size for `/api/chat`; `null` leaves the server default |
| `vllm.base_url` / `nvidia.base_url` | OpenAI-compatible endpoint used by `--vllm` / `--nvidia` |
| `vllm.api_key` / `nvidia.api_key` | bearer key for those endpoints (any placeholder is fine if the proxy needs none) |
| `vllm.model` / `nvidia.model` | model id used when that flag is passed without `--model` |
| `vllm.enable_thinking` / `nvidia.enable_thinking` | `false` asks the template to skip the reasoning trace; `null` sends no chat-template kwargs at all |

A key of the `ollama:` block that is absent (or `null`) is simply not sent, so a
server that predates one of those fields keeps working: a 400 on the request is
answered by resending it without the extra field, the same graceful fallback the
`--vllm` path gives `chat_template_kwargs`.

`model.name` is **not** read here - the CLI takes `--model` (or `vllm.model` /
`nvidia.model`), so editing that key changes nothing for this harness.

Precedence everywhere: command line > `config.yaml` > built-in default
(`http://127.0.0.1:3111` for the game, `300` s for the LLM timeout, `easy` for
the preset, `1` for the seed).

## How a run works

`farm_agent/cli.py:main` is the whole pipeline; each step is one module:

1. **Config** - `config.load_config()` reads `config.yaml`, `parse_args` fills the
   defaults above and records `args.backend` (`ollama`, `vllm` or `nvidia`);
   `--vllm` / `--nvidia` without `--model` take `vllm.model` / `nvidia.model`.
2. **Fail fast** - `GET /api/v1/health`; if the server is down the run exits 1
   *before* a log file or a reset, so no half-written artefact is left behind.
   The health payload (version, preset, seed, episode, tick, browser clients) is
   printed and logged: it is the proof of *which* server this log describes.
3. **Goal check** - `--list-objectives` prints the catalogue and exits. Otherwise
   `objectives.validate_objective` rejects an unknown id (exit 2) *before*
   anything is reset, because the server silently falls back to `starter_easy`
   for an id it does not know.
4. **Served constants** - `GET /api/v1/config` gives the crop economics, preset
   values, `minutes_per_tick` and `ticks_per_day`, so the prompt quotes the
   numbers this build actually scores instead of stale literals.
5. **Logging** - the stem and the `.log` file (`config.setup_logging`).
6. **Episode** - `episode.prepare_episode` posts
   `{preset, seed, objective?, task?}` to `/api/v1/reset` (or keeps the running
   one with `--no-reset`, resuming it if it was paused) and then verifies that the
   episode really scores the expected objective - a mismatch aborts with exit 3.
7. **Tools** - `GET /api/v1/tools` becomes the Ollama/OpenAI tool schemas. A tool
   the server advertises without a local wrapper is a hard error, so the model
   can never be offered something the harness cannot execute.
8. **Loop** - `agent.run_agent` (Ollama) or `agent.run_agent_openai` (`--vllm`,
   `--nvidia`); all of them use the same prompt, tools, nudges and stop reasons.
9. **Reporting** - `reporting.summarise_run` re-reads the state, prints the points,
   writes `_latency.png` and `_summary.json`, and `--reset-after` starts a fresh
   episode.

## The loop, step by step

`messages` starts as `[system prompt, task]`. Every iteration of
`while i < max_steps`:

1. `runtime.STEPS_USED = i` - the observation the model gets back in this step
   quotes the steps it has left (`budget:` line).
2. One chat completion with the tool schemas. Ollama answers
   `message.tool_calls` plus `prompt_eval_count` / `eval_count` and the
   `*_duration` fields; vLLM answers `choices[0].message` plus `usage`. Both token
   counts are summed into the summary, and every turn is printed and logged
   (`Content:` and `Thinking:` when the model emits them).
3. The assistant turn is appended to the history, then the turn's tool calls are
   run by their wrappers in `tools.py`, in the order the model asked for them. One
   call is one `POST /api/v1/action` (`include_state=1`) and its answer is the JSON
   the model reads:
   `{"status": "true"|"false", "action", "message", "data", "reason"?, "note"?}`.
   `data` is the server's own payload (`litres`, `moisture`, `steps`, ...) and
   `note` only appears when the harness changed the request (a clipped `advance`).
4. Bookkeeping (`tools.apply_result`, one call per action in call order):
   successful calls go into `tool_calls` with the time the server took for that
   call (`time_taken` - the round trip, not the LLM step), `visited` / `jumps` /
   `revisits` follow the tile after every action, and every failure is counted per
   `reason` in `invalid_move_object`. A call the harness refuses locally (an
   unknown `move` direction, a bad argument, an invented tool name) is booked the
   same way without spending a round trip.

Three guards keep a stuck agent from burning the whole budget silently:

| guard | trigger | what happens |
| --- | --- | --- |
| `NO_PROGRESS` | the server says `ok` but the farmer moved 0 tiles (`move_to` out of energy) | the reply is rewritten to `status: "false"`, `reason: NO_PROGRESS` with the fix ("call `rest`") |
| stuck | 3 identical tool calls in a row (`runtime.STUCK_REPEATS`) | one user message names the repeated call and lists the alternatives (`rest`, `move`, `inspect`) |
| silent step | a turn with no tool call while the objective is open | a user message asks for exactly one tool; after 2 such nudges the episode ends with `stop_reason: silent_steps` |

The episode ends as soon as the action that completes the objective comes back -
the server re-evaluates the goal inside every action, so the winning `harvest`
already answers `is_complete: true` and the loop stops on the same step.

## What the model sees

* **System prompt** (`prompt.py`) - the task, ~18 `IMPORTANT` rules, a `WORLD`
  block (size, tick length, landmarks, rocks, the crop table with `yield_kg` /
  `sell_price` / seasons from `/api/v1/config`, can size, silo, rovers), the
  objective with its progress and reward, the objective arithmetic
  (`served.objective_plan_to_text`: how much is still missing and which crop
  closes it), the `CURRENT STATE` digest, and the tool catalogue with the
  docstrings from `tools.py`.
* **Every tool reply** - the JSON above; the next turn's `status` line is
  authoritative, a failure always carries a machine-readable `reason`.
* **Every observation** (`rendering.state_to_text`): day/season/clock, weather,
  player tile, energy, money, can/tank/silo, seeds, landmarks, prices, objective
  progress + reward, the `budget:` line (steps left, days to the deadline,
  advance cap), rovers, an ASCII field grid with the worked plots, and the last
  events.

## Module map

| module | responsibility |
| --- | --- |
| `runtime` | the mutable run state (state, counters, budgets) - bottom of the dependency graph, nothing imports another module from it |
| `config` | `config.yaml`, argparse, the log file and the artefact stem |
| `api` | `GET` / `POST` against farm-sim, `refresh_state`, the `server:` line |
| `tracking` | visited tiles, jumps, revisits, the per-episode reset |
| `budget` | tick/minute maths, `budget_to_text`, `clamp_advance` |
| `rendering` | the `CURRENT STATE` block and the field grid |
| `objectives` | catalogue, validation, post-reset verification |
| `episode` | reset / adopt, `resolve_task` |
| `served` | the served constants the prompt may quote |
| `tools` | 18 action wrappers and the tool schemas sent to the model |
| `prompt` | the system message and tool-argument decoding |
| `agent` | the LLM loop, once per transport (`run_agent` for Ollama, `run_agent_openai` for vLLM and NVIDIA NIM) |
| `check_api` | the scripted `--check-api` run |
| `reporting` | latency plot, statistics, `_summary.json` |
| `cli` | `main()` |

## Three backends, one loop

| | Ollama (default) | `--vllm` / `--nvidia` |
| --- | --- | --- |
| client | `ollama.Client(host=ollama_url(), timeout=args.llm_timeout)` | `openai.OpenAI(base_url=openai_base_url(backend), api_key=openai_api_key(backend))` |
| host | `server_urls.ollama_url`, can be a remote box | `vllm.base_url` (DGX proxy) or `nvidia.base_url` (hosted NIM), picked by `args.backend` |
| reply | `response.message` (content, `thinking`, `tool_calls`) | `choices[0].message` |
| tool result | `{"role": "tool", "content": ...}` | same, plus the `tool_call_id` the OpenAI schema requires |
| assistant echo | the `ChatResponse.message` object is appended as-is | rebuilt as a plain dict, because the OpenAI SDK message objects do not round-trip |
| tokens | `prompt_eval_count` / `eval_count` | `usage.prompt_tokens` / `completion_tokens` |
| history | unbounded within `--max-steps` | trimmed to `MAX_HISTORY_MESSAGES = 42` messages by `trim_history`, keeping the system prompt, the task and whole tool groups |

The vLLM deployment itself must allow tool calls; the harness cannot invent that:

```bash
vllm serve Qwen/Qwen3.8-27B-FP8 \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --port 8000
```

```yaml
# config.yaml
vllm:
  base_url: "http://dgx-proxy.example:8000/v1"
  api_key: "EMPTY"
  model: "Qwen/Qwen3.8-27B-FP8"
  enable_thinking: false
```

`enable_thinking: false` is sent as `extra_body={"chat_template_kwargs":
{"enable_thinking": false}}`; if the served template does not know the flag the
request is retried once without it, so an older deployment still runs (it just
keeps the reasoning trace, which costs hundreds of tokens per step).
`enable_thinking: null` sends no chat-template kwargs at all.

The NVIDIA block has the same shape, because the hosted API is the same API - so
`--nvidia` is the same loop with a different `base_url` / `api_key` / `model`:

```yaml
# config.yaml
nvidia:
  base_url: "https://integrate.api.nvidia.com/v1"
  api_key: "nvapi-..."
  model: "openai/gpt-oss-20b"
  enable_thinking: false
```

`GET /v1/models` with that key lists the ids it may call, and only the ones whose
model card lists **tool calling** can drive this harness - there is no fallback for
a model that will not emit `tool_calls`. The block that ships here was checked
against the live endpoint: `openai/gpt-oss-20b` accepts `chat_template_kwargs`,
returns `tool_calls` with `arguments` as a JSON string, and a 2-turn episode
recorded `no_of_llm_calls: 2`, `tool_calls_sent: 2` and `llm_backend: nvidia`. Two
practical notes: hosted NIM can be slow (the first turn took 211 s for 3.2 k input
tokens, the second 67 s), so raise `--llm-timeout`; and no Qwen build is served,
which is why `nvidia.model` defaults to gpt-oss while the `vllm:` block keeps Qwen.
`python ../experiments/fake_model_smoke.py` drives all three backends with a stub
client, so a block that stops resolving is caught without a model server.

## Outputs

Every run writes three artefacts into `--log-folder` (default
`../experiments/test_logs_baseline_farm_sim`), named after the stem built in
`config.py`:

```
<model>_<preset>_seed<seed><tag>.log           full transcript: prompt, every
                                               LLM reply, every tool reply
<model>_<preset>_seed<seed><tag>_latency.png   seconds per successful tool call
<model>_<preset>_seed<seed><tag>_summary.json  everything a table needs
```

The `model` part is sanitised (`:`, `.`, `/`, `\` become `_`), and the objective is
**not** part of the name: the same model, preset and seed replaying a different
objective overwrites the previous artefacts - pass `--tag` for such a sweep.

`_summary.json` keys (written by `reporting.summarise_run`):

| key | meaning |
| --- | --- |
| `model`, `llm_backend`, `base_url`, `preset`, `seed`, `task`, `days_played`, `tick` | what was run, on which server (`llm_backend`: `ollama`, `vllm` or `nvidia`) |
| `objective_requested`, `objective` | what was asked for vs. what the episode reports (`id`, `text`, `deadline_day`, `reward`, `progress`, `is_complete`, `is_failed`) |
| `metrics` | the server's counters **and `score`** - see [Scoring](#scoring-how-a-run-is-judged) |
| `points_gained` | `metrics.score` - the number to put in a results table |
| `points_gained_object` | `objective_completed`, `harvests`, `kg_harvested`, `kg_sold`, `earned`, `spent`, `crops_died` |
| `stop_reason` | `objective_complete`, `objective_failed`, `max_steps`, `silent_steps`, `llm_error` (and `check_api_pass` / `check_api_fail` for `--check-api`) |
| `no_of_llm_calls`, `no_of_tool_calls`, `tool_calls_sent`, `tool_calls_per_turn` | one chat completion per model **turn** (`no_of_llm_calls`), the tool calls that succeeded (`no_of_tool_calls`), every call the model asked for (`tool_calls_sent`) and the average a turn carried; a `--check-api` run has `0` turns |
| `total_input_tokens`, `total_output_tokens`, `time_taken_stats` | cost of the episode (`mean`/`median`/`variance`/`std_dev`/`min`/`max` seconds per successful call) |
| `tool_calls`, `tool_calls_deduped`, `step_order` | what the agent actually did; `step_order` = index of the first `till`, `plant_crop`, `collect_water`, `water`, `harvest`, `sell` |
| `invalid_moves`, `invalid_move_object`, `invalid_move_rate`, `jumps`, `revisits`, `player_positions` | harness-side bookkeeping, not the sim's own `invalid_actions` |

Exit codes: `0` finished (whatever the stop reason), `1` the simulation could not
be reached, `2` unknown `--objective`, `3` the episode scores a different
objective than the run claims.

## Scoring: how a run is judged

There is no reward model and no extra judge - the simulation scores the episode
and the harness only reports it. `points_gained` is `metrics.score`; the formula
lives in the simulation's `sim/metrics.js`, and the per-term explanation, the
shaped `reward` on every action and the objective bonus are documented in
[`../farm_sim/README.md` → Scoring](../farm_sim/README.md#scoring-the-points-an-agent-earns).
In short:

```
score = earned + 8*harvests + 0.5*kg_harvested + 50*objective_completed
        - 0.25*invalid_actions - 2*crops_died
```

Only selling counts as `earned`, the objective bonus is paid in money and enters
the score through `objective_completed`, and steps spent on `inspect` or on
rejected moves cost points without ever gaining any. `--check-api` therefore
prints a small but non-zero `points_gained`: it plants, harvests and sells.

## Troubleshooting

| symptom | cause / fix |
| --- | --- |
| `cannot reach the farm-sim server at http://127.0.0.1:3111` (exit 1) | the server is on 3100 (a bare `npm start`); run `PORT=3111 npm start` or pass `--url` |
| `--check-api` exits 1 | read the failing step it prints; usually no seed to plant (money) or water still missing |
| `stop_reason: llm_error` | the model did not answer within `--llm-timeout` (default 300 s, after two retries on a timeout / dropped connection) - raise it, or check `ollama list` / the vLLM proxy / NIM (the hosted endpoint can take minutes per turn). If it is the *first* turn of a model that was idle, Ollama was reloading the weights: set `ollama.keep_alive` in `config.yaml` |
| vLLM answers 400 about `tool_calls` | the server was started without `--enable-auto-tool-choice --tool-call-parser ...` |
| NIM answers 404/400 for the model | `nvidia.model` is not an id `GET /v1/models` lists for that key, or its model card does not list tool calling; the id in `config.yaml` is only a default and `--model` overrides it |
| `objective mismatch: the episode scores X, this run expects Y` (exit 3) | `--no-reset` adopted a leftover episode, or `--objective` names a goal the preset does not default to - drop `--no-reset` or pass a matching `--objective` |
| `stop_reason: silent_steps` | the model keeps answering in prose instead of calling a tool; a bigger model or a stricter `--task` helps |
| the log of an earlier run is gone | same model + preset + seed and no `--tag` (the objective is not in the stem) |
| `Tool X not found` in the transcript | the model invented a tool name; `GET /api/v1/tools` is the only catalogue |
