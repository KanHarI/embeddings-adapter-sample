from embeddings_adapter.data.openai_cache import get_embeddings


def test_embedding() -> None:
    embedding = get_embeddings("Hello world")
    assert len(embedding) == 1536
