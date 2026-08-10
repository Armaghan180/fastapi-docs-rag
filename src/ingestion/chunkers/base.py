"""Shared types and helpers used by every chunking strategy.

Both chunkers need to (a) know where markdown headings are without getting confused by
lines that merely start with "#" inside a fenced code block (Python comments, shell
comments, mkdocs-material's "# (1)!" annotation markers), and (b) resolve each heading to
the same anchor MkDocs would generate, so citations link to the exact section on
https://fastapi.tiangolo.com instead of just the page.
"""

import re
from dataclasses import dataclass, field

import tiktoken

import config

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
HEADING_ID_RE = re.compile(r"\s*\{\s*#([\w-]+)\s*\}\s*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")

# Approximate tokenizer shared by both chunkers for sizing decisions. Exact token count
# depends on the actual embedding/generation model, but cl100k_base is close enough to size
# chunks consistently -- what matters is comparing strategies against each other, not
# matching any one model's tokenizer exactly.
_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text))


@dataclass
class Chunk:
    id: str
    doc_path: str  # relative to docs/en/docs, e.g. "tutorial/first-steps.md"
    heading_path: list[str]  # ancestor headings, e.g. ["Advanced User Guide", "Events"]
    anchor: str | None
    text: str
    strategy: str
    index: int  # position of this chunk within its document
    token_count: int = field(init=False)
    url: str = field(init=False)

    def __post_init__(self):
        self.token_count = count_tokens(self.text)
        self.url = doc_url(self.doc_path, self.anchor)


def slugify(heading_text: str) -> str:
    """Approximates MkDocs' auto-generated heading slug for headings with no explicit {#id}."""
    text = re.sub(r"`([^`]*)`", r"\1", heading_text)
    text = re.sub(r"\*\*?([^*]*)\*\*?", r"\1", text)
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "-", text)
    return text.strip("-")


def parse_heading_line(line: str) -> tuple[int, str, str] | None:
    """Returns (level, display_text, anchor) if the line is a heading, else None."""
    match = HEADING_RE.match(line)
    if not match:
        return None
    level = len(match.group(1))
    raw_text = match.group(2).strip()

    id_match = HEADING_ID_RE.search(raw_text)
    if id_match:
        anchor = id_match.group(1)
        display_text = HEADING_ID_RE.sub("", raw_text).strip()
    else:
        anchor = slugify(raw_text)
        display_text = raw_text

    return level, display_text, anchor


def build_clean_lines_and_heading_state(
    markdown: str,
) -> tuple[list[str], list[tuple[list[str], str | None]]]:
    """Walks the document once, fence-aware, and returns:

    - clean_lines: the doc's lines with heading `{ #id }` attr-list cruft rebuilt away
    - heading_state: for each line, the (heading_path, anchor) of the section it belongs to

    Both chunkers need this: heading_based uses it to find section boundaries, and
    fixed_size uses it purely to attach citation metadata to windows that don't respect
    those boundaries.
    """
    clean_lines: list[str] = []
    heading_state: list[tuple[list[str], str | None]] = []
    stack: list[tuple[int, str]] = []
    current_anchor: str | None = None
    in_fence = False

    for line in markdown.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
            clean_lines.append(line)
        elif not in_fence:
            parsed = parse_heading_line(line)
            if parsed:
                level, text, anchor = parsed
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, text))
                current_anchor = anchor
                clean_lines.append("#" * level + " " + text)
            else:
                clean_lines.append(line)
        else:
            clean_lines.append(line)

        heading_state.append(([text for _, text in stack], current_anchor))

    return clean_lines, heading_state


def token_counts(lines: list[str]) -> list[int]:
    return [count_tokens(line) + 1 for line in lines]  # +1 approximates the newline


def sliding_windows(n: int, tok: list[int], size_tokens: int, overlap_tokens: int) -> list[tuple[int, int]]:
    """Greedily packs line indices [0, n) into token-budgeted windows with overlap.

    A window always contains at least one line even if that line alone exceeds the
    budget (e.g. one huge inlined code block), so this can never stall.
    Returns inclusive (start, end) line-index pairs.
    """
    windows: list[tuple[int, int]] = []
    start = 0
    while start < n:
        end = start
        total = 0
        while end < n and (total == 0 or total + tok[end] <= size_tokens):
            total += tok[end]
            end += 1
        end -= 1
        windows.append((start, end))

        if end >= n - 1:
            break

        overlap = 0
        back = end
        while back > start and overlap < overlap_tokens:
            overlap += tok[back]
            back -= 1
        start = back + 1

    return windows


def doc_url(doc_path: str, anchor: str | None) -> str:
    path = doc_path[:-3] if doc_path.endswith(".md") else doc_path
    if path == "index":
        path = ""
    elif path.endswith("/index"):
        path = path[: -len("index")]
    else:
        path = path + "/"

    url = f"{config.FASTAPI_DOCS_SITE_URL}/{path}"
    if anchor:
        url = f"{url}#{anchor}"
    return url


class Chunker:
    strategy_name: str

    def chunk(self, doc_path: str, markdown: str) -> list[Chunk]:
        raise NotImplementedError
