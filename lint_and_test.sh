black embeddings_adapter
isort --profile black embeddings_adapter
pip install -e .
flake8 embeddings_adapter
mypy --strict embeddings_adapter
pytest ./embeddings_adapter/tests
