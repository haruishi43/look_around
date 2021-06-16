#!/usr/bin/env python3

from torch.utils.data import Dataset as _Dataset


class Dataset(_Dataset):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>: initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>: return the size of dataset.
    -- <__getitem__>: get a data point.
    """

    def __init__(self):
        pass

    def __len__(self):
        """Return the total number of images in the dataset."""
        return NotImplementedError

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        raise NotImplementedError

    @property
    def num_classes(self) -> int:
        raise NotImplementedError
