#!/usr/bin/env python3

import os
from typing import Callable, Generic, List, TypeVar

from .episode import Episode

T = TypeVar("T", bound=Episode)


class Dataset(Generic[T]):
    r"""Base class for dataset specification."""
    episodes: List[T]

    @staticmethod
    def scene_from_scene_path(scene_path: str) -> str:
        r"""Helper method to get the scene name from an episode.
        :param scene_path: The path to the scene, assumes this is formatted
                            ``/path/to/<scene_name>.<ext>``
        :return: <scene_name> from the path
        """
        return os.path.splitext(os.path.basename(scene_path))[0]

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        r"""Returns a list of scene names that would be loaded with this dataset.
        Useful for determining what scenes to split up among different workers.
        :param config: The config for the dataset
        :return: A list of scene names that would be loaded with the dataset
        """
        assert cls.check_config_paths_exist(config)  # type: ignore[attr-defined]
        dataset = cls(config)  # type: ignore[call-arg]
        return list(map(cls.scene_from_scene_path, dataset.scene_ids))

    @classmethod
    def build_content_scenes_filter(cls, config) -> Callable[[T], bool]:
        """Returns a filter function that takes an episode and returns True if that
        episode is valid under the CONTENT_SCENES field of the provided config
        """
        scenes_to_load = set(config.CONTENT_SCENES)

        def _filter(ep: T) -> bool:
            return (
                ALL_SCENES_MASK in scenes_to_load
                or cls.scene_from_scene_path(ep.scene_id) in scenes_to_load
            )

        return _filter

    @property
    def num_episodes(self) -> int:
        r"""number of episodes in the dataset"""
        return len(self.episodes)

    @property
    def scene_ids(self) -> List[str]:
        r"""unique scene ids present in the dataset."""
        return sorted({episode.scene_id for episode in self.episodes})

    def get_scene_episodes(self, scene_id: str) -> List[T]:
        r"""..
        :param scene_id: id of scene in scene dataset.
        :return: list of episodes for the :p:`scene_id`.
        """
        return list(
            filter(lambda x: x.scene_id == scene_id, iter(self.episodes))
        )

    def get_episodes(self, indexes: List[int]) -> List[T]:
        r"""..
        :param indexes: episode indices in dataset.
        :return: list of episodes corresponding to indexes.
        """
        return [self.episodes[episode_id] for episode_id in indexes]

    def get_episode_iterator(self, *args: Any, **kwargs: Any) -> Iterator:
        r"""Gets episode iterator with options. Options are specified in
        :ref:`EpisodeIterator` documentation.
        :param args: positional args for iterator constructor
        :param kwargs: keyword args for iterator constructor
        :return: episode iterator with specified behavior
        To further customize iterator behavior for your :ref:`Dataset`
        subclass, create a customized iterator class like
        :ref:`EpisodeIterator` and override this method.
        """
        return EpisodeIterator(self.episodes, *args, **kwargs)

    def to_json(self) -> str:
        class DatasetJSONEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()

                return (
                    obj.__getstate__()
                    if hasattr(obj, "__getstate__")
                    else obj.__dict__
                )

        result = DatasetJSONEncoder().encode(self)
        return result

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        r"""Creates dataset from :p:`json_str`.
        :param json_str: JSON string containing episodes information.
        :param scenes_dir: directory containing graphical assets relevant
            for episodes present in :p:`json_str`.
        Directory containing relevant graphical assets of scenes is passed
        through :p:`scenes_dir`.
        """
        raise NotImplementedError

    def filter_episodes(self, filter_fn: Callable[[T], bool]) -> "Dataset":
        r"""Returns a new dataset with only the filtered episodes from the
        original dataset.
        :param filter_fn: function used to filter the episodes.
        :return: the new dataset.
        """
        new_episodes = []
        for episode in self.episodes:
            if filter_fn(episode):
                new_episodes.append(episode)
        new_dataset = copy.copy(self)
        new_dataset.episodes = new_episodes
        return new_dataset

    def get_splits(
        self,
        num_splits: int,
        episodes_per_split: Optional[int] = None,
        remove_unused_episodes: bool = False,
        collate_scene_ids: bool = True,
        sort_by_episode_id: bool = False,
        allow_uneven_splits: bool = False,
    ) -> List["Dataset"]:
        r"""Returns a list of new datasets, each with a subset of the original
        episodes.
        :param num_splits: the number of splits to create.
        :param episodes_per_split: if provided, each split will have up to this
            many episodes. If it is not provided, each dataset will have
            :py:`len(original_dataset.episodes) // num_splits` episodes. If
            max_episodes_per_split is provided and is larger than this value,
            it will be capped to this value.
        :param remove_unused_episodes: once the splits are created, the extra
            episodes will be destroyed from the original dataset. This saves
            memory for large datasets.
        :param collate_scene_ids: if true, episodes with the same scene id are
            next to each other. This saves on overhead of switching between
            scenes, but means multiple sequential episodes will be related to
            each other because they will be in the same scene.
        :param sort_by_episode_id: if true, sequences are sorted by their
            episode ID in the returned splits.
        :param allow_uneven_splits: if true, the last splits can be shorter
            than the others. This is especially useful for splitting over
            validation/test datasets in order to make sure that all episodes
            are copied but none are duplicated.
        :return: a list of new datasets, each with their own subset of
            episodes.
        All splits will have the same number of episodes, but no episodes will
        be duplicated.
        """
        if self.num_episodes < num_splits:
            raise ValueError(
                "Not enough episodes to create those many splits."
            )

        if episodes_per_split is not None:
            if allow_uneven_splits:
                raise ValueError(
                    "You probably don't want to specify allow_uneven_splits"
                    " and episodes_per_split."
                )

            if num_splits * episodes_per_split > self.num_episodes:
                raise ValueError(
                    "Not enough episodes to create those many splits."
                )

        new_datasets = []

        if episodes_per_split is not None:
            stride = episodes_per_split
        else:
            stride = self.num_episodes // num_splits
        split_lengths = [stride] * num_splits

        if allow_uneven_splits:
            episodes_left = self.num_episodes - stride * num_splits
            split_lengths[:episodes_left] = [stride + 1] * episodes_left
            assert sum(split_lengths) == self.num_episodes

        num_episodes = sum(split_lengths)

        rand_items = np.random.choice(
            self.num_episodes, num_episodes, replace=False
        )
        if collate_scene_ids:
            scene_ids: Dict[str, List[int]] = {}
            for rand_ind in rand_items:
                scene = self.episodes[rand_ind].scene_id
                if scene not in scene_ids:
                    scene_ids[scene] = []
                scene_ids[scene].append(rand_ind)
            rand_items = []
            list(map(rand_items.extend, scene_ids.values()))
        ep_ind = 0
        new_episodes = []
        for nn in range(num_splits):
            new_dataset = copy.copy(self)  # Creates a shallow copy
            new_dataset.episodes = []
            new_datasets.append(new_dataset)
            for _ii in range(split_lengths[nn]):
                new_dataset.episodes.append(self.episodes[rand_items[ep_ind]])
                ep_ind += 1
            if sort_by_episode_id:
                new_dataset.episodes.sort(key=lambda ep: ep.episode_id)
            new_episodes.extend(new_dataset.episodes)
        if remove_unused_episodes:
            self.episodes = new_episodes
        return new_datasets
