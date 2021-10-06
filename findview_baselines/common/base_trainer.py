#!/usr/bin/env python3

from copy import deepcopy
from collections import defaultdict
import os
from os import PathLike
from typing import Any, Dict, List, Optional

from gym import spaces
import numpy as np
import torch

from LookAround.config import Config
from LookAround.core.spaces import ActionSpace
from LookAround.FindView.vec_env import VecEnv, construct_envs


class BaseTrainer(object):
    """Generic trainer class that serves as a base template for more
    specific trainer classes like RL trainer

    Includes only the most basic functionality
    """

    # Properties
    cfg: Config
    num_updates_done: int
    num_steps_done: int

    # Hidden Properties
    _base_cfg: Config
    _flush_secs: int
    _last_checkpoint_percent: float

    def __init__(self) -> None:

        # Initialize properties
        self.num_updates_done = 0
        self.num_steps_done = 0
        self._flush_secs = 30
        self._last_checkpoint_percent = -1.0

    @property
    def flush_secs(self):
        return self._flush_secs

    @flush_secs.setter
    def flush_secs(self, value: int):
        self._flush_secs = value

    @property
    def ckpt_dir(self) -> PathLike:
        raise NotImplementedError

    @property
    def tb_dir(self) -> PathLike:
        raise NotImplementedError

    @property
    def video_dir(self) -> PathLike:
        raise NotImplementedError

    def train(self) -> None:
        raise NotImplementedError

    def save_checkpoint(
        self,
        file_name: str,
        save_state: Dict[str, Any],
        extra_state: Optional[Dict[str, Any]] = None
    ) -> None:
        raise NotImplementedError

    def load_checkpoint(
        self,
        checkpoint_path: PathLike,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """Load checkpoint of specified path as a dict.
        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args
        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)


class BaseRLTrainer(BaseTrainer):
    """Base trainer class for RL trainers"""

    # Properties
    envs: Optional[VecEnv]
    device: torch.device

    def __init__(
        self,
        cfg: Config,
    ) -> None:
        super().__init__()

        # initialize properties
        self.cfg = cfg
        self._base_cfg = cfg.base_trainer
        self.envs = None

        # set spaces to None
        self._obs_space = None
        self._policy_action_space = None

        # Some sanity checks in the input config
        if self._base_cfg.num_updates != -1 and self._base_cfg.total_num_steps != -1:
            raise RuntimeError(
                "`num_updates` and `total_num_steps` are both specified. One must be -1.\n"
                "`num_updates`: {} `total_num_steps`: {}".format(
                    self._base_cfg.num_updates, self._base_cfg.total_num_steps,
                )
            )
        if self._base_cfg.num_updates == -1 and self._base_cfg.total_num_steps == -1:
            raise RuntimeError(
                "One of `num_updates` and `total_num_steps` must be specified.\n"
                "`num_updates`: {} `total_num_steps`: {}".format(
                    self._base_cfg.num_updates, self._base_cfg.total_num_steps,
                )
            )
        if self._base_cfg.num_ckpts != -1 and self._base_cfg.ckpt_interval != -1:
            raise RuntimeError(
                "`num_ckpts` and `ckpt_interval` are both specified."
                "  One must be -1.\n"
                " `num_ckpts`: {} `ckpt_interval`: {}".format(
                    self._base_cfg.num_ckpts, self._base_cfg.ckpt_interval
                )
            )
        if self._base_cfg.num_ckpts == -1 and self._base_cfg.ckpt_interval == -1:
            raise RuntimeError(
                "One of `num_ckpts` and `ckpt_interval` must be specified"
                " `num_ckpts`: {} `ckpt_interval`: {}".format(
                    self._base_cfg.num_ckpts, self._base_cfg.ckpt_interval
                )
            )

    def percent_done(self) -> float:
        if self._base_cfg.num_updates != -1:
            return self.num_updates_done / self._base_cfg.num_updates
        else:
            return self.num_steps_done / self._base_cfg.total_num_steps

    def is_done(self) -> bool:
        return self.percent_done() >= 1.0

    def should_checkpoint(self) -> bool:
        needs_checkpoint = False
        if self._base_cfg.num_ckpts != -1:
            checkpoint_every = 1 / self._base_cfg.num_ckpts
            if (
                self._last_checkpoint_percent + checkpoint_every
                < self.percent_done()
            ):
                needs_checkpoint = True
                self._last_checkpoint_percent = self.percent_done()
        else:
            needs_checkpoint = (
                self.num_updates_done % self._base_cfg.ckpt_interval
            ) == 0

        return needs_checkpoint

    @property
    def obs_space(self) -> spaces.Dict:

        # NOTE: obtain obs_space dynamically
        if self._obs_space is None and self.envs is not None:
            self._obs_space = self.envs.observation_spaces[0]

        return self._obs_space

    @obs_space.setter
    def obs_space(self, new_obs_space: spaces.Dict):
        self._obs_space = new_obs_space

    @property
    def policy_action_space(self) -> ActionSpace:

        # NOTE: obtain action_space dynamically
        if self._policy_action_space is None and self.envs is not None:
            self._policy_action_space = self.envs.action_spaces[0]

        return self._policy_action_space

    @policy_action_space.setter
    def policy_action_space(self, new_policy_action_space: ActionSpace):
        self._policy_action_space = new_policy_action_space

    @property
    def ckpt_dir(self) -> PathLike:
        ckpt_dir = self._base_cfg.ckpt_dir.format(
            results_root=self.cfg.results_root,
            run_id=str(self._base_cfg.run_id),
        )
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    @property
    def tb_dir(self) -> PathLike:
        tb_dir = self._base_cfg.tb_dir.format(
            tb_root=self.cfg.tb_root,
            run_id=str(self._base_cfg.run_id),
        )
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir, exist_ok=True)
        return tb_dir

    @property
    def video_dir(self) -> PathLike:
        video_dir = self._base_cfg.video_dir.format(
            results_root=self.cfg.results_root,
            run_id=str(self._base_cfg.run_id),
        )
        if not os.path.exists(video_dir):
            os.makedirs(video_dir, exist_ok=True)
        return video_dir

    def save_checkpoint(
        self,
        file_name: str,
        save_state: Dict[str, Any],
        extra_state: Optional[Dict[str, Any]] = None
    ) -> None:
        """Save checkpoint with specified name.
        Args:
            file_name: file name for checkpoint
        Returns:
            None
        """

        if extra_state is not None:
            save_state["extra_state"] = extra_state

        torch.save(
            save_state, os.path.join(self.ckpt_dir, file_name)
        )

    def _init_rlenvs(
        self,
        split: str = "train",
        cfg: Optional[Config] = None,
    ) -> None:
        if cfg is None:
            cfg = Config(deepcopy(self.cfg))
        split_cfg = getattr(cfg, split)
        assert split_cfg is not None

        if split_cfg.dtype == "torch.float32":
            dtype = torch.float32
        elif split_cfg.dtype == "torch.float64":
            dtype = torch.float64
        else:
            raise ValueError()

        self.envs = construct_envs(
            cfg=cfg,
            split=split,
            is_rlenv=True,
            dtype=dtype,
            device=torch.device(split_cfg.device),
            vec_type=split_cfg.vec_type,
        )

    METRICS_BLACKLIST = [
        "episode_id",
        "initial_rotation",
        "target_rotation",
        "current_rotation",
        "steps_for_shortest_path",
    ]

    @classmethod
    def _extract_scalars_from_info(
        cls,
        info: Dict[str, Any],
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls,
        infos: List[Dict[str, Any]],
    ) -> Dict[str, List[float]]:
        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results
