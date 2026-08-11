"""Pull FastAPI's docs from GitHub at a pinned commit and resolve them into plain markdown.

FastAPI's docs are built with MkDocs Material and use a custom snippet-embedding syntax:
a "code block" in the source markdown is often not literal code, it's a directive like

    {* ../../docs_src/body/tutorial001_py310.py hl[2] *}
    {* ../../fastapi/openapi/docs.py ln[9:24] hl[18:24] *}

that tells the MkDocs build to pull in another file (optionally a specific line range) at
render time. If we chunked the raw markdown as-is, every code example would silently
disappear -- a huge part of what makes these docs useful. So this module resolves each
directive by fetching the referenced source and inlining it as a real fenced code block
before anything gets chunked.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import requests

import config

GITHUB_API = "https://api.github.com"
RAW_BASE = "https://raw.githubusercontent.com"

# Matches: {* ../../<path> [ln[..]] [hl[..]] [title["..."]] *}
SNIPPET_RE = re.compile(
    r"\{\*\s*\.\./\.\./(?P<path>\S+?)"
    r"(?:\s+ln\[(?P<ln>[^\]]+)\])?"
    r"(?:\s+hl\[[^\]]*\])?"
    r"(?:\s+title\[\"(?P<title>[^\"]*)\"\])?"
    r"\s*\*\}"
)

FENCE_LANG_BY_EXT = {".py": "python", ".json": "json", ".yml": "yaml", ".yaml": "yaml"}

# MkDocs Material's "termy" extension wraps animated terminal examples in a <div class="termy">
# and colors each line with <font color="#hex"> / <span style="..."> for the fake-terminal
# effect. That's presentational noise once the HTML can't actually render (arbitrary hex codes
# mixed into "console" text), so it's stripped down to plain text. This is intentionally
# narrow -- it does NOT touch other HTML tags, since several doc pages legitimately show raw
# HTML (<form>, <button>, <html>...) as code examples inside fences, and those must survive
# untouched.
TERMY_DIV_RE = re.compile(r'<div class="termy">\s*\n(.*?)\n\s*</div>', re.DOTALL)
FONT_TAG_RE = re.compile(r"<font[^>]*>(.*?)</font>", re.DOTALL)
SPAN_STYLE_RE = re.compile(r'<span style="[^"]*">(.*?)</span>', re.DOTALL)


def clean_termy_markup(markdown: str) -> str:
    markdown = TERMY_DIV_RE.sub(lambda m: m.group(1), markdown)
    markdown = FONT_TAG_RE.sub(lambda m: m.group(1), markdown)
    markdown = SPAN_STYLE_RE.sub(lambda m: m.group(1), markdown)
    return markdown


@dataclass
class DocPage:
    repo_path: str  # full path in the repo, e.g. "docs/en/docs/tutorial/first-steps.md"
    relative_path: str  # path relative to docs/en/docs, e.g. "tutorial/first-steps.md"
    content: str


def get_doc_paths(session: requests.Session, repo: str, commit: str) -> list[str]:
    """List every markdown path under FASTAPI_DOCS_SUBPATH at the pinned commit, minus
    the auto-generated API reference stubs and underscore-prefixed meta files."""
    url = f"{GITHUB_API}/repos/{repo}/git/trees/{commit}"
    resp = session.get(url, params={"recursive": "1"}, timeout=30)
    resp.raise_for_status()
    tree = resp.json()["tree"]

    prefix = config.FASTAPI_DOCS_SUBPATH + "/"
    paths = []
    for entry in tree:
        path = entry["path"]
        if entry["type"] != "blob" or not path.startswith(prefix) or not path.endswith(".md"):
            continue
        rel = path[len(prefix):]
        basename = rel.rsplit("/", 1)[-1]
        if basename.startswith(config.EXCLUDED_DOC_NAME_PREFIX):
            continue
        if any(rel.startswith(p) for p in config.EXCLUDED_DOC_PREFIXES):
            continue
        paths.append(path)
    return sorted(paths)


def fetch_raw(session: requests.Session, repo: str, commit: str, path: str) -> str:
    url = f"{RAW_BASE}/{repo}/{commit}/{path}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def _parse_line_ranges(spec: str) -> list[tuple[int, int]]:
    """Parse an ln[] spec like "1:2,12:16,29,38:41" into [(1,2), (12,16), (29,29), (38,41)]."""
    ranges = []
    for part in spec.split(","):
        part = part.strip()
        if ":" in part:
            start, end = part.split(":")
            ranges.append((int(start), int(end)))
        else:
            ranges.append((int(part), int(part)))
    return ranges


def _slice_lines(text: str, ranges: list[tuple[int, int]]) -> str:
    """Extract 1-indexed inclusive line ranges, joining non-contiguous gaps with a marker."""
    lines = text.splitlines()
    pieces = []
    prev_end = None
    for start, end in ranges:
        if prev_end is not None and start > prev_end + 1:
            pieces.append("# ...")
        pieces.extend(lines[start - 1:end])
        prev_end = end
    return "\n".join(pieces)


def resolve_snippets(markdown: str, session: requests.Session, repo: str, commit: str, cache: dict) -> str:
    """Replace every {* ... *} snippet directive with a real fenced code block."""

    def _replace(match: re.Match) -> str:
        path = match.group("path")
        if path not in cache:
            cache[path] = fetch_raw(session, repo, commit, path)
        source = cache[path]

        ln_spec = match.group("ln")
        code = _slice_lines(source, _parse_line_ranges(ln_spec)) if ln_spec else source.rstrip("\n")

        ext = "." + path.rsplit(".", 1)[-1] if "." in path else ""
        lang = FENCE_LANG_BY_EXT.get(ext, "")

        title = match.group("title")
        title_comment = f"# {title}\n" if title else ""

        return f"```{lang}\n{title_comment}{code}\n```"

    return SNIPPET_RE.sub(_replace, markdown)


def fetch_all_docs() -> tuple[list[DocPage], dict]:
    """Fetch and resolve every doc page. Returns (pages, manifest)."""
    session = requests.Session()
    repo, commit = config.FASTAPI_REPO, config.FASTAPI_DOCS_COMMIT
    prefix = config.FASTAPI_DOCS_SUBPATH + "/"

    doc_paths = get_doc_paths(session, repo, commit)
    snippet_cache: dict[str, str] = {}

    pages = []
    for repo_path in doc_paths:
        raw = fetch_raw(session, repo, commit, repo_path)
        resolved = resolve_snippets(raw, session, repo, commit, snippet_cache)
        cleaned = clean_termy_markup(resolved)
        pages.append(DocPage(repo_path=repo_path, relative_path=repo_path[len(prefix):], content=cleaned))

    manifest = {
        "repo": repo,
        "commit": commit,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "page_count": len(pages),
        "snippet_files_resolved": len(snippet_cache),
    }
    return pages, manifest


def save_docs(pages: list[DocPage], manifest: dict) -> None:
    config.DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    for page in pages:
        out_path = config.DATA_RAW_DIR / page.relative_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(page.content, encoding="utf-8")

    manifest_path = config.DATA_RAW_DIR / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    print(f"Fetching docs from {config.FASTAPI_REPO}@{config.FASTAPI_DOCS_COMMIT[:12]}...")
    pages, manifest = fetch_all_docs()
    save_docs(pages, manifest)
    print(f"Saved {len(pages)} pages ({manifest['snippet_files_resolved']} source files inlined) "
          f"to {config.DATA_RAW_DIR}")


if __name__ == "__main__":
    main()
