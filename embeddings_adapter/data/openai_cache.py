import typing

import openai

openai.api_key = "YOUR_API_KEY"


def get_embeddings(input: str) -> list[float]:
    response = openai.Embedding.create(  # type: ignore
        input=[input],
        model="text-embedding-ada-002",
    )["data"][0]["embedding"]
    return typing.cast(list[float], response)
