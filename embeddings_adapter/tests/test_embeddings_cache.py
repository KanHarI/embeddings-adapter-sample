from embeddings_adapter.data.openai_cache import EmbeddingsCache


def test_embeddings_cache() -> None:
    embeddings_cache = EmbeddingsCache()
    embeddings_cache.load_from_json()
    embedding = embeddings_cache.get("Hello world")
    assert len(embedding) == 1536
    embeddings_cache.save_to_json()
