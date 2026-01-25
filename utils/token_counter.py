import tiktoken

def count_openai_tokens(text: str, model: str = "gpt-4o") -> int:
    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except KeyError:
        encoding = tiktoken.encoding_for_model("gpt-4")
        return len(encoding.encode(text))
