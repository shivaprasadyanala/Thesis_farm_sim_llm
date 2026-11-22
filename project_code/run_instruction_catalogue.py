"""Entry point of the instruction-catalogue batch (the batch's `baseline_farm_sim`).

Run from ``project_code``, exactly like the single-episode script:

    python run_instruction_catalogue.py --model all --dry-run
    python run_instruction_catalogue.py --vllm --model qwen3:8b --repeats 3

``python -m instruction_catalogue`` runs the same ``main``; this file exists so
the batch is started from the same folder, and named the same way, as
``agentic_approach_baseline_farm_sim.py``.

Everything else (flags, matrix order, artefacts) is documented in
``instruction_catalogue/README.md``.
"""

import sys

from instruction_catalogue.cli import main

if __name__ == "__main__":
    sys.exit(main())
