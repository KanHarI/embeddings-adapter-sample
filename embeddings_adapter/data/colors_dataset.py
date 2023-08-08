import random

import torch.utils.data

from embeddings_adapter.data.colors_raw_data import ColorsDataset
from embeddings_adapter.data.openai_cache import EmbeddingsCache, GLOBAL_EMBEDDINGS_CACHE


class ColorsDataloader(torch.utils.data.Dataset[torch.Tensor]):
    def __init__(self, colors_dataset: ColorsDataset) -> None:
        super().__init__()
        self.colors_dataset = colors_dataset
        self.embeddings_cache = GLOBAL_EMBEDDINGS_CACHE
        self.embeddings_cache.load_from_json()

    def __len__(self) -> int:
        return 1_000_000_000  # A hack, read the torch documentation

    def __getitem__(self, index: int) -> torch.Tensor:
        # Sample randomly, every time 2 samples from the same category and another 2 from different category
        selected_two_pair_first_category = random.randint(
            0, len(self.colors_dataset) - 1
        )
        selected_different_category = random.randint(0, len(self.colors_dataset) - 2)
        if selected_different_category >= selected_two_pair_first_category:
            selected_different_category += 1
        sampled_1_first_category = random.randint(
            0,
            len(self.colors_dataset[selected_two_pair_first_category].category_items)
            - 1,
        )
        sampled_2_first_category = random.randint(
            0,
            len(self.colors_dataset[selected_two_pair_first_category].category_items)
            - 2,
        )
        if sampled_2_first_category >= sampled_1_first_category:
            sampled_2_first_category += 1
        sampled_different_category = random.randint(
            0,
            len(self.colors_dataset[selected_different_category].category_items) - 1,
        )
        sampled_1_embeddings = self.embeddings_cache.get(
            self.colors_dataset[selected_two_pair_first_category].category_items[
                sampled_1_first_category
            ]
        )
        sampled_2_embeddings = self.embeddings_cache.get(
            self.colors_dataset[selected_two_pair_first_category].category_items[
                sampled_2_first_category
            ]
        )
        sampled_3_embeddings = self.embeddings_cache.get(
            self.colors_dataset[selected_different_category].category_items[
                sampled_different_category
            ]
        )
        return torch.Tensor(
            [
                sampled_1_embeddings,
                sampled_2_embeddings,
                sampled_3_embeddings,
            ]
        )
