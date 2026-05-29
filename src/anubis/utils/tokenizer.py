import tiktoken


enc = tiktoken.get_encoding("o200k_base")


def count_tokens(text: str) -> int:
    return len(enc.encode(text))


def split_text_by_token_budget(text: str, max_tokens_per_chunk: int) -> list[str]:
    """Split ``text`` into segments each at most ``max_tokens_per_chunk`` tokens."""
    if max_tokens_per_chunk <= 0:
        return [text]
    ids = enc.encode(text)
    if not ids:
        return []
    out: list[str] = []
    i = 0
    while i < len(ids):
        chunk_ids = ids[i : i + max_tokens_per_chunk]
        out.append(enc.decode(chunk_ids))
        i += max_tokens_per_chunk
    return out