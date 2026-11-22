"""Instruction-catalogue experiment runner: task x directness level x prompt phrasing.

The single-episode harness in ``farm_agent`` answers "how well does one model do
on one goal". This package answers the thesis question around it: *does the way
the instruction is phrased change the outcome* - same world, same goal, only the
wording differs. The catalogue of phrasings already existed as a plain data file
(``../Indirect_instructions.py``); this package turns it into a batch of episodes
and collects the per-episode summaries into one table.

Nothing here modifies the harness. Every episode is one subprocess of the
unchanged entry script (``agentic_approach_baseline_farm_sim.py``), which keeps
the existing artefact semantics - one ``.log``, one ``_summary.json``, one
``_latency.png`` per episode - and means a batch crash can never leave the
harness in a half-initialised state.

Module map::

    instructions   the catalogue: load, validate, task -> objective/preset
                   mapping, trial expansion (model x task x level x prompt x repeat)
    cli            argparse of the batch, ``--model all`` / ``--preset default``
                   resolution, models.yaml, the matrix preview (--dry-run)
    runner         pre-flight (health, objectives, served models), one
                   subprocess per trial, --resume/--limit/--smoke/--fail-fast
    results        the per-episode _summary.json files, aggregated into
                   results[model][task][level] plus a long-format CSV

Documented invocation (run from ``project_code``)::

    python -m instruction_catalogue --model all --dry-run
    python -m instruction_catalogue --model all --vllm --repeats 3
    python run_instruction_catalogue.py --model qwen3:8b --tasks harvest_crop_task \
           --levels indirect,implicit --preset normal --repeats 3

``python -m`` puts ``project_code`` on ``sys.path``, which is what makes the
``farm_agent`` imports of the child process and of this package resolve.
"""

__all__ = [
    "cli",
    "instructions",
    "results",
    "runner",
]
