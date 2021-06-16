#!/usr/bin/env python3

from collections import defaultdict
import copy
import random
from typing import Optional

import numpy as np

from torch.utils.data.sampler import RandomSampler, Sampler, SequentialSampler

from tqdm import tqdm

AVAI_SAMPLERS = [
    "RandomSceneCategorySampler",
    "SequentialSampler",
    "RandomSampler",
]

__all__ = [
    "build_sampler",
]


"""FIXME:

RandomSceneCatagorySampler reduces the number of batches by quite a lot.
I need to take a closer look at what categories and the number of instances
we are receiving from this sampler.

I'm guessing it's due to having low number of classes as well as low number of
instances for some of the categories

"""


class RandomSceneCategorySampler(Sampler):
    """Randomly samples N identities each with K instances.

    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        batch_size (int): batch size.
        num_instances (int): number of instances per identity in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        assert (
            batch_size >= num_instances
        ), f"batch_size={batch_size} must be no less than num_instances={num_instances}"

        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_categories_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        for index, data in enumerate(tqdm(self.data_source)):
            # get category_id
            category_id = data["category"]
            self.index_dic[category_id].append(index)
        self.category_ids = list(self.index_dic.keys())

        # estimate number of examples in an epoch
        # TODO: improve precision
        self.length = 0
        for category_id in tqdm(self.category_ids):
            idxs = self.index_dic[category_id]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for category_id in self.category_ids:
            idxs = copy.deepcopy(self.index_dic[category_id])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(
                    idxs, size=self.num_instances, replace=True
                )
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[category_id].append(batch_idxs)
                    batch_idxs = []

        avai_category_ids = copy.deepcopy(self.category_ids)
        final_idxs = []

        while len(avai_category_ids) >= self.num_categories_per_batch:
            selected_ids = random.sample(
                avai_category_ids, self.num_categories_per_batch
            )
            for category_id in selected_ids:
                batch_idxs = batch_idxs_dict[category_id].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[category_id]) == 0:
                    avai_category_ids.remove(category_id)

        return iter(final_idxs)

    def __len__(self):
        return self.length


def build_sampler(
    data_source: list,
    train_sampler: str,
    batch_size: int = 32,
    num_instances: int = 4,
    **kwargs,
) -> Optional[Sampler]:
    """Builds a training sampler.
    Args:
        data_source (list): contains tuples of (img_path(s), pid, camid).
        train_sampler (str): sampler name (default: ``RandomSampler``).
        batch_size (int, optional): batch size. Default is 32.
        num_instances (int, optional): number of instances per identity in a
            batch (when using ``RandomIdentitySampler``). Default is 4.
    """
    assert (
        train_sampler in AVAI_SAMPLERS
    ), f"train_sampler must be one of {AVAI_SAMPLERS}, but got {train_sampler}"
    print(f"*** Using {train_sampler}")

    if train_sampler == "RandomSceneCategorySampler":
        sampler = RandomSceneCategorySampler(
            data_source, batch_size, num_instances
        )

    elif train_sampler == "SequentialSampler":
        sampler = SequentialSampler(data_source)

    elif train_sampler == "RandomSampler":
        sampler = RandomSampler(data_source)

    return sampler
