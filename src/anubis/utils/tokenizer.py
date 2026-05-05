import tiktoken


enc = tiktoken.get_encoding("o200k_base")

def count_tokens(text: str) -> int:
    return len(enc.encode(text))