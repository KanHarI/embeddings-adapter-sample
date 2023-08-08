from setuptools import setup

__VERSION__ = "0.1.0"

setup(
    name="embeddings_adapter",
    version=__VERSION__,
    packages=["embeddings_adapter"],
    python_requires=">=3.10",
    install_requires=[
        "dacite",
        "einops",
        "hydra-core",
        "numpy",
        "openai[embeddings,wandb]",
        "torch",
        "torchaudio",
        "torchvision",
        "tqdm",
        "types-tqdm",
        "wandb",
    ],
    include_package_data=True,
)
