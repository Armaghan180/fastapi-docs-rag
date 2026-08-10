"""Primary chunker: splits on markdown headings so each chunk is a coherent section --
a heading plus its own content, never mixed with a sibling section's. A section that's
still too large (e.g. a long tutorial page under one heading) falls back to the same
token-budgeted windowing fixed_size uses, so no single chunk is unbounded.
"""

import config

from . import base


class HeadingBasedChunker(base.Chunker):
    strategy_name = "heading_based"

    def chunk(self, doc_path: str, markdown: str) -> list[base.Chunk]:
        clean_lines, heading_state = base.build_clean_lines_and_heading_state(markdown)
        sections = self._split_into_sections(clean_lines, heading_state)

        chunks = []
        index = 0
        for heading_path, anchor, section_lines in sections:
            text = "\n".join(section_lines).strip()
            if not text:
                continue

            tok = base.token_counts(section_lines)
            if sum(tok) <= config.MAX_SECTION_TOKENS:
                chunks.append(self._make_chunk(doc_path, heading_path, anchor, text, index))
                index += 1
                continue

            for start, end in base.sliding_windows(
                len(section_lines), tok, config.MAX_SECTION_TOKENS, config.FIXED_CHUNK_OVERLAP_TOKENS
            ):
                sub_text = "\n".join(section_lines[start : end + 1]).strip()
                if not sub_text:
                    continue
                chunks.append(self._make_chunk(doc_path, heading_path, anchor, sub_text, index))
                index += 1

        return chunks

    def _make_chunk(self, doc_path, heading_path, anchor, text, index) -> base.Chunk:
        return base.Chunk(
            id=f"{doc_path}::{self.strategy_name}::{index}",
            doc_path=doc_path,
            heading_path=heading_path,
            anchor=anchor,
            text=text,
            strategy=self.strategy_name,
            index=index,
        )

    @staticmethod
    def _split_into_sections(
        clean_lines: list[str],
        heading_state: list[tuple[list[str], str | None]],
    ) -> list[tuple[list[str], str | None, list[str]]]:
        """Groups consecutive lines that share the same (heading_path, anchor) into sections."""
        sections = []
        current_key = None
        current_lines: list[str] = []

        for line, (heading_path, anchor) in zip(clean_lines, heading_state):
            key = (tuple(heading_path), anchor)
            if key != current_key:
                if current_lines:
                    sections.append((list(current_key[0]), current_key[1], current_lines))
                current_key = key
                current_lines = []
            current_lines.append(line)

        if current_lines:
            sections.append((list(current_key[0]), current_key[1], current_lines))

        return sections
