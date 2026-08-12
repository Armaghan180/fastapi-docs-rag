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
# Pinned to a specific commit (not "master") so the ingested corpus, the eval set's
# ground-truth answers, and citation URLs all stay in sync even after the docs change upstream.
FASTAPI_REPO = os.getenv("FASTAPI_REPO", "fastapi/fastapi")
FASTAPI_DOCS_COMMIT = os.getenv("FASTAPI_DOCS_COMMIT", "244d66308d6c525f394d0c2ce32dabceb2ed262b")
FASTAPI_DOCS_SUBPATH = "docs/en/docs"  # English docs only, markdown source

# Skip auto-generated API reference stubs (mkdocstrings "::: fastapi.X" directives with
# almost no prose) and underscore-prefixed meta/test files that aren't real doc content.
EXCLUDED_DOC_PREFIXES = ("reference/",)
EXCLUDED_DOC_NAME_PREFIX = "_"

# --- API / UI ---
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

# --- Models ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
GENERATION_MODEL = "gpt-4o-mini"
JUDGE_MODEL = "gpt-4o-mini"

# --- Chunking ---
FIXED_CHUNK_SIZE_TOKENS = 400
FIXED_CHUNK_OVERLAP_TOKENS = 50

# heading_based: sections larger than this get sub-split (with the same overlap/window
# logic as fixed_size) so one long tutorial page can't become a single giant chunk.
MAX_SECTION_TOKENS = 800

FASTAPI_DOCS_SITE_URL = "https://fastapi.tiangolo.com"

# --- Retrieval ---
TOP_K = 5
# hybrid: each of dense/BM25 retrieves this many candidates before Reciprocal Rank Fusion
# trims down to TOP_K. RRF_K=60 is the standard constant from Cormack et al. (2009).
CANDIDATE_POOL_SIZE = 20
RRF_K = 60

# --- Strategy matrix (the eval compares these configs) ---
# Each entry: (chunking_strategy, retrieval_method)
STRATEGIES = [
    ("fixed_size", "dense"),
    ("heading_based", "dense"),
    ("heading_based", "hybrid"),
]
