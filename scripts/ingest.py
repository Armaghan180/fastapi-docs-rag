"""CLI entrypoint: fetch FastAPI's docs, then chunk + embed + index them under every strategy.

Usage: python scripts/ingest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ingestion import build_index, fetch_docs


def main() -> None:
    fetch_docs.main()
    build_index.main()


if __name__ == "__main__":
    main()
