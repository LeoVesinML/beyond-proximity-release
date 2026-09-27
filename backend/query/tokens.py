"""Token counting for query-mode benchmarks."""
from __future__ import annotations


def count_tokens(text: str, model: str = "cl100k_base") -> int:
    """Count tokens in *text*. Uses tiktoken when available, else chars/4."""
    if not text:
        return 0
    try:
        import tiktoken
        enc = tiktoken.get_encoding(model)
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)
