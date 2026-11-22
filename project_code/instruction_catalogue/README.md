# instruction_catalogue - the prompt-wording batch

Runs `Indirect_instructions.py` (three task groups x `direct/indirect/implicit` x
three phrasings) as one episode per cell and aggregates the per-episode summaries.

One episode = the *unchanged* single-episode harness
(`agentic_approach_baseline_farm_sim.py`, i.e. `farm_agent`) as a subprocess. The
batch only decides **which** episodes exist, starts them, and collects their
`_summary.json` files.

## Quick start

```powershell
cd project_code
$env:PORT=3111; npm start          # in ../farm_sim, in another terminal

python -m instruction_catalogue --model all --dry-run     # print the matrix
python -m instruction_catalogue --smoke                   # 3 scripted episodes, no LLM
python -m instruction_catalogue --model qwen3:8b --repeats 3
python -m instruction_catalogue --vllm --model Qwen/Qwen3.8-27B --tasks harvest_crop_task
```

`python run_instruction_catalogue.py ...` from the same folder runs the same `main`.

## The matrix

```
model  ->  task group  ->  level  ->  prompt phrasing  ->  repeat
```

* `--model` a model id, `all` (the `<backend>:` list of `models.yaml`), or omitted
  for `--vllm`/`--nvidia` (then `config.yaml` decides, as for one episode).
* `--tasks` default `all` = the catalogue's three groups.
* `--levels` default `all` = `direct, indirect, implicit`.
* `--repeats` default 1. Repeat `r` uses seed `--seed + r - 1`, and **all** phrasings
  of one cell share the same repeat seeds - direct/indirect/implicit are compared on
  identical worlds (paired design), not on differently seeded farms.

Default goal/world per task group (`--preset`/`--objective-map` override):

| task group                  | preset | objective      |
|-----------------------------|--------|----------------|
| `plant_crop_water_task`     | easy   | `starter_easy` |
| `harvest_crop_task`         | normal | `harvest_normal` |
| `harvest_crop_earn_money_task` | hard | `profit_hard` |

Prompts are passed **verbatim** via `--task` (they are data, not prompts to
rewrite); `--task` only changes the instruction text - the server scores the
objective, which is why the child verifies `--objective` after the reset.

## Artefacts

One sub-folder per model, named after the sanitised model id (`/`, `:`, `.` ->
`_`, exactly as the harness renames the artefact files themselves):

```
<log-folder>/
  Qwen_Qwen3_8-27B/                     <- one folder per model
    Qwen_Qwen3_8-27B_normal_seed2_harvest_crop_task_direct_p2_r2.log
                                                _summary.json
                                                _latency.png
                                                _llm_latency.png
  _catalogue_<backend>_seed<seed><tag>_summary.json
  _catalogue_<backend>_seed<seed><tag>.csv
```

Per episode, exactly what one hand-started episode writes, now inside its model's
folder. `--flat-logs` puts every model directly into `<log-folder>` instead (the
old layout); two ids that sanitise to the same folder name are refused up front.

Per batch, in the root:

* `_catalogue_<backend>_seed<seed><tag>_summary.json` - `results[model][task][level]`
  with the per-episode values **as lists** (a 0/100/0 split stays visible) plus
  derived means, and a flat `cells` list. `summaries` names are relative to the
  batch folder (`Qwen_Qwen3_8-27B/<stem>_summary.json`), and `group_models`
  records the layout.
* `_catalogue_<backend>_seed<seed><tag>.csv` - long format, one row per episode
  (model, task, level, prompt index, repeat, seed, points, completed, stop reason,
  LLM/tool calls, seconds per LLM request, seconds per tool call, seconds per
  episode, days played, summary path, prompt text).
* `--smoke` names them `_catalogue_smoke_...`, so a scripted check never overwrites
  a real aggregate.

The aggregate stays in the root on purpose: it spans every model of the batch.

Episodes that were never attempted (`--limit`, `--resume` holes) count as
`missing`, not as zero.

## Latency metrics

Three clocks per episode, never interchangeable:

| summary field / CSV column | scope | what it times |
|---|---|---|
| `llm_time_taken_stats.mean` / `llm_time_taken_mean` | one **LLM request** | client-side wall clock of one chat completion (network + queueing + prefill + decode). One entry per model turn, so `len(llm_time_taken) == no_of_llm_calls`; a retry inside `chat_completion` (template kwarg rejected with 400) is still one turn. |
| `time_taken_stats.mean` / `time_taken_mean` | one **tool call** | round trip of one *successful* `POST /api/v1/action` to the farm server. Invalid moves are counted in `invalid_move_object`, never here. |
| `episode_time` / `episode_time` | the **whole agent loop** | every model turn plus every action request, from the first `client.chat` to the last tool reply. Excludes the health check, the episode reset, the tool-schema fetch, the summary write and the plots. |

`episode_time_per_llm_call` and `episode_time_per_tool_call` divide that same total
by the two counts. The per-cell means (`mean_llm_time_taken`, `mean_time_taken`,
`mean_episode_time`) average the per-episode values over the repeats, and the
console prints one table per scope. `<stem>_latency.png` plots the action
latencies, `<stem>_llm_latency.png` the model latencies.

**Not comparable with the legacy baseline.** In
`experiments/test_logs_baseline_farm_sim/gemma4_26b_*` the same field was booked as
*LLM step + tool execution* - which is why it reads ~3 s where these episodes read
~0.01 s. Only episodes written by this harness may be compared on these columns.

## Flags

| flag | meaning |
|------|---------|
| `--dry-run` | print the expanded matrix (episode number, model, task, level, prompt index, repeat, seed, preset/objective, prompt) and exit - no server, no LLM |
| `--smoke` | one scripted `--check-api` episode per task group: reset + objective verification + artefact naming + aggregation without model time |
| `--list-models` | curated list of every backend in `models.yaml` + what the endpoint now serves |
| `--list-tasks` | task groups, their levels, prompt counts and resolved preset/objective |
| `--resume` | skip episodes whose `_summary.json` exists (an interrupted batch continues where it stopped) |
| `--limit N` | run at most `N` episodes (a cheap prefix of the matrix) |
| `--fail-fast` | stop at the first episode that did not really run |
| `--trial-timeout S` | wall-clock cap per episode (`0` = off) |
| `--tag` | suffix for the aggregate file names |
| `--flat-logs` | one flat folder instead of one sub-folder per model (`--resume` follows the layout either way) |
| `--url`, `--vllm`, `--nvidia` | as for one episode (default: Ollama, `config.yaml`) |
| `--max-steps`, `--llm-timeout`, `--max-advance-ticks` | passed to every episode |
| `--preset`, `--objective-map` | world / goal overrides (`default` = the table above) |
| `--instructions-file`, `--models-file` | another catalogue / model list, same shape |

Exit codes: `0` every attempted episode ran, `1` at least one did not,
`2` a selection problem (unknown task/level/objective, unreadable file, bad flag).

"Did not run" is stricter than the child's exit code: an episode the backend
answered with `llm_error`, or one that exited 0 without writing a `_summary.json`,
is reported as a failure - a model without tool support answers all 297 prompts
with `llm_error`, and that must not look like a finished experiment. The aggregate
is written either way; the CSV keeps such rows (`stop_reason = llm_error`).

## Pre-flight (before episode 1)

1. farm-sim reachable (`/api/v1/health`), server preset/seed printed, note when no
   browser is watching.
2. Every objective of the matrix exists in the server's catalogue - the server
   silently replaces an unknown id with `starter_easy`, so a typo has to stop here.
   Deadline and reward are printed per objective.
3. The models are reported by the backend (`/api/tags` or `/v1/models`) - a
   warning only, but it catches a renamed Ollama tag before a long night.

## Module map

| module | role |
|--------|------|
| `instructions.py` | catalogue loading/validation, `Trial`, matrix expansion, `--dry-run` output - no I/O beyond reading the file |
| `cli.py` | argparse, `models.yaml`, previews, exit codes |
| `runner.py` | pre-flight, one subprocess per episode, resume/limit bookkeeping |
| `results.py` | `_summary.json` -> `results[model][task][level]`, console tables, JSON + CSV |

## A new catalogue file

`--instructions-file` takes any file with the same shape as
`Indirect_instructions.py`:

```python
tasks = [{"some_task": {"direct": ["..."], "indirect": ["..."], "implicit": ["..."]}}]
```

A new task name needs a goal/world mapping: add it to
`instructions.TASK_OBJECTIVES`, or pass `--objective-map some_task=<objective id>`
and `--preset easy|normal|hard`. Unknown groups/levels/objectives fail before the
first episode instead of shortening the matrix silently.

