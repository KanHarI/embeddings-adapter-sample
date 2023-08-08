import json
import os
import typing

import openai

openai.api_key = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")


class EmbeddingsCache:
    def __init__(self) -> None:
        self.cache = {}

    def get(self, input: str) -> list[float]:
        if input not in self.cache:
            self.cache[input] = get_embeddings(input)
        return typing.cast(list[float], self.cache[input])

    def save_to_json(self) -> None:
        with open("embeddings_cache.json", "w") as f:
            json.dump(self.cache, f)

    def load_from_json(self) -> None:
        if not os.path.exists("embeddings_cache.json"):
            return
        with open("embeddings_cache.json", "r") as f:
            self.cache = json.load(f)


def get_embeddings(input: str) -> list[float]:
    response = openai.Embedding.create(  # type: ignore
        input=[input],
        model="text-embedding-ada-002",
    )["data"][0]["embedding"]
    return typing.cast(list[float], response)
