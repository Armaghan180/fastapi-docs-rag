"""CLI entrypoint: run the eval question set against every strategy config in config.STRATEGIES.

Usage: python scripts/evaluate.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval import run_eval

if __name__ == "__main__":
    run_eval.main()
