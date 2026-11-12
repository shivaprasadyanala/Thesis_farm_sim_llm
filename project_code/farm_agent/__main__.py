"""`python -m farm_agent ...` - the same entry point as the thin script."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
