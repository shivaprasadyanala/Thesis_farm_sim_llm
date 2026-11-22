"""``python -m instruction_catalogue`` - same entry point as the root script."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
