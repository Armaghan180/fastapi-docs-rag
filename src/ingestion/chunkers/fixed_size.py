"""Baseline chunker: a token-budgeted sliding window over the document, with no awareness
of section boundaries. This is the naive approach heading_based is measured against --
a window can start mid-section or straddle a heading, splitting an explanation from the
code example that follows it.
"""

import config

from . import base


class FixedSizeChunker(base.Chunker):
    strategy_name = "fixed_size"

    def chunk(self, doc_path: str, markdown: str) -> list[base.Chunk]:
        clean_lines, heading_state = base.build_clean_lines_and_heading_state(markdown)
        tok = base.token_counts(clean_lines)
        windows = base.sliding_windows(
            len(clean_lines), tok, config.FIXED_CHUNK_SIZE_TOKENS, config.FIXED_CHUNK_OVERLAP_TOKENS
        )

        chunks = []
        for index, (start, end) in enumerate(windows):
            text = "\n".join(clean_lines[start : end + 1]).strip()
            if not text:
                continue
            heading_path, anchor = heading_state[start]
            chunks.append(
                base.Chunk(
                    id=f"{doc_path}::{self.strategy_name}::{index}",
                    doc_path=doc_path,
                    heading_path=heading_path,
                    anchor=anchor,
                    text=text,
                    strategy=self.strategy_name,
                    index=index,
                )
            )
        return chunks
