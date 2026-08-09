"""Central config: models, paths, and the strategy matrix used across ingestion/retrieval/eval."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
ROOT_DIR = Path(__file__).parent
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
CHROMA_DIR = ROOT_DIR / "data" / "chroma"
EVAL_QUESTIONS_PATH = ROOT_DIR / "src" / "eval" / "questions.json"
EVAL_RESULTS_DIR = ROOT_DIR / "src" / "eval" / "results"

# --- Source docs ---
FASTAPI_REPO = os.getenv("FASTAPI_REPO", "tiangolo/fastapi")
FASTAPI_DOCS_COMMIT = os.getenv("FASTAPI_DOCS_COMMIT", "master")
FASTAPI_DOCS_SUBPATH = "docs/en/docs"  # English docs only, markdown source

# --- Models ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"

# --- Chunking ---
FIXED_CHUNK_SIZE_TOKENS = 400
FIXED_CHUNK_OVERLAP_TOKENS = 50

# --- Retrieval ---
TOP_K = 5
HYBRID_DENSE_WEIGHT = 0.5  # used in reciprocal rank fusion

# --- Strategy matrix (the eval compares these configs) ---
# Each entry: (chunking_strategy, retrieval_method)
STRATEGIES = [
    ("fixed_size", "dense"),
    ("heading_based", "dense"),
    ("heading_based", "hybrid"),
]
