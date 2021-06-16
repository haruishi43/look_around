#!/usr/bin/env python3

import os.path as osp
import logging
from typing import Any, Dict, List, Optional

import pandas as pd

from PIL import Image

from pers2pano.data.datasets.dataset import Dataset
from pers2pano.data.datasets.registry import DATASET_REGISTRY


@DATASET_REGISTRY.register(name="SUN360")
class SUN360(Dataset):
    def __init__(
        self,
        root_path: str,
        data_path: str,
        img_transforms: Optional[Any] = None,
        **kwargs,
    ) -> None:
        """SUN360 Dataset

        args:
            root_path (str): root directory of the dataset
            data_path (str): path to `csv` file
            img_transforms (Any): transforms for images
        """
        assert osp.exists(root_path), f"{root_path} is not a valid path"
        assert osp.isfile(data_path), f"{data_path} is not a valid path"
        self.root = root_path
        self.df = pd.read_csv(data_path)

        category = osp.dirname(data_path).split("/")[-1]
        if category == "all":
            category = ["indoor", "outdoor"]
        if isinstance(category, str):
            category = [category]

        if len(self.df["category"].unique()) > 1:
            logger = logging.getLogger("pers2pano")
            logger.warning("Using multiple categories!")
            raise ValueError

        classes = []
        for _category in ("indoor", "outdoor", "other"):
            if _category in category:
                if _category in ("indoor", "outdoor"):
                    sub_category_list = self.get_sub_categories(_category)
                    for sub in sub_category_list:
                        classes.append(f"{_category}_{sub}")
                else:
                    # other
                    classes.append(_category)
        assert len(classes) > 0
        self.classes = classes

        self.img_transforms = img_transforms

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        data = self.df.iloc[idx]

        # Prep image
        img_path = data["img_path"]
        img = Image.open(osp.join(self.root, img_path))
        if self.img_transforms is not None:
            img = self.img_transforms(img)

        # Prep categories
        category = data["category"]
        sub_category = data["sub_category"]
        if category in ("indoor", "outdoor"):
            _class = f"{category}_{sub_category}"
        else:
            _class = category

        class_id = self.classes.index(_class)

        return {
            "category_name": category,
            "sub_category_name": sub_category,
            "img": img,
            "category": class_id,
            "img_path": img_path,
        }

    def __len__(self) -> int:
        return len(self.df)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def get_all_sub_categories(self) -> Dict[str, List[str]]:
        """Get all sub categories for the whatever category you are using

        returns:
            Dict[str, List[str]]
        """
        # FIXME: hardcoded category txt file is located here
        categories = self.df["category"].unique()

        all_sub_categories = {}
        for category in categories:
            if category == "others":
                continue

            # FIXME: hardcoded category txt file that contains all categories
            read_sub_categories = self.get_sub_categories(category=category)

            df = self.df[self.df["category"] == category]
            sub_categories = df["sub_category"].unique()
            assert all(sub in read_sub_categories for sub in sub_categories)
            all_sub_categories[category] = sub_categories

        return all_sub_categories

    @staticmethod
    def get_sub_categories(category: str) -> List[str]:
        """Get sub categories from predefined text file"""
        assert category in ("indoor", "outdoor")
        dir_path = osp.dirname(osp.realpath(__file__))
        txt_file = osp.join(dir_path, f"{category}.txt")
        assert osp.isfile(txt_file), f"{txt_file} doesn't exist!"
        sub_categories = open(txt_file).read().splitlines()
        return sub_categories
