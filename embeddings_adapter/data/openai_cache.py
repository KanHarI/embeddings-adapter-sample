import hashlib
import json
import os
import typing

import openai

openai.api_key = os.environ.get("OPENAI_API_KEY", "YOUR_API_KEY")

CURRENT_FILE_PATH = os.path.dirname(os.path.abspath(__file__))


class EmbeddingsCache:
    def __init__(self) -> None:
        self.cache: dict[str, list[float]] = {}

    def get(self, input: str) -> list[float]:
        input_hash = hashlib.sha256(input.encode("utf-8")).hexdigest()
        if input_hash not in self.cache:
            self.cache[input_hash] = get_embeddings(input)
        return self.cache[input_hash]

    def save_to_json(self) -> None:
        path_to_json = os.path.join(CURRENT_FILE_PATH, "embeddings_cache.json")
        with open(path_to_json, "w") as f:
            json.dump(self.cache, f)

    def load_from_json(self) -> None:
        path_to_json = os.path.join(CURRENT_FILE_PATH, "embeddings_cache.json")
        if not os.path.exists(path_to_json):
            return
        with open(path_to_json, "r") as f:
            self.cache = json.load(f)


def get_embeddings(input: str) -> list[float]:
    response = openai.Embedding.create(  # type: ignore
        input=[input],
        model="text-embedding-ada-002",
    )["data"][0]["embedding"]
    return typing.cast(list[float], response)


GLOBAL_EMBEDDINGS_CACHE = EmbeddingsCache()
