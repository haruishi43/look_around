#!/usr/bin/env python3

from collections import defaultdict
import os
from os import PathLike
from typing import Any, Dict, List, Optional, Tuple, Union

from gym import spaces
import numpy as np
import torch

from LookAround.config import Config
from LookAround.core import logger
from LookAround.core.spaces import ActionSpace
from LookAround.FindView.vec_env import VecEnv

from findview_baselines.common.tensorboard_utils import TensorboardWriter
from findview_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)


class BaseTrainer(object):
    """Generic trainer class that serves as a base template for more
    specific trainer classes like RL trainer

    Includes only the most basic functionality
    """

    # properties
    cfg: Config
    num_updates_done: int
    num_steps_done: int

    # hidden properties
    _flush_secs: int
    _last_checkpoint_percent: float

    def __init__(self) -> None:
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

    def eval(self) -> None:
        """Main method of trainer evaluation. Calls _eval_checkpoint() that
        is specified in Trainer class that inherits from BaseRLTrainer
        or BaseILTrainer
        Returns:
            None

        NOTE: make sure that `num_envs=1` inorder to get the most consistent results
        """

        logger.add_filehandler(
            self.cfg.log_file.format(
                split="test",
                log_root=self.cfg.log_root,
                run_id=self.cfg.run_id,
            )
        )

        self.device = torch.device(self.cfg.test.device)

        with TensorboardWriter(
            self.tb_dir,
            flush_secs=self.flush_secs,
        ) as writer:
            ckpt_path = os.path.join(self.ckpt_dir, self.cfg.test.ckpt_path)
            if os.path.isfile(ckpt_path):
                # evaluate singe checkpoint
                proposed_index = get_checkpoint_id(ckpt_path)
                assert proposed_index is not None, \
                    f"ERR: could not find valid ckpt for {ckpt_path}"
                ckpt_idx = proposed_index
                self._eval_checkpoint(
                    ckpt_path,
                    writer,
                    checkpoint_index=ckpt_idx,
                )
            else:
                # evaluate multiple checkpoints in order
                prev_ckpt_ind = -1
                while True:
                    current_ckpt = poll_checkpoint_folder(
                        self.ckpt_dir, prev_ckpt_ind
                    )
                    if current_ckpt is None:
                        break
                    logger.info(f"=======current_ckpt: {current_ckpt}=======")
                    prev_ckpt_ind += 1
                    self._eval_checkpoint(
                        checkpoint_path=current_ckpt,
                        writer=writer,
                        checkpoint_index=prev_ckpt_ind,
                    )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
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

    # properties
    envs: Optional[VecEnv]
    device: torch.device

    def __init__(
        self,
        cfg: Config,
    ) -> None:
        super().__init__()

        # FIXME: probably change this to trainer cfg or something
        self.cfg = cfg
        self.envs = None

        # set spaces to None
        self._obs_space = None
        self._policy_action_space = None

        # Some sanity checks in the input config
        if cfg.num_updates != -1 and cfg.total_num_steps != -1:
            raise RuntimeError(
                "`num_updates` and `total_num_steps` are both specified. One must be -1.\n"
                "`num_updates`: {} `total_num_steps`: {}".format(
                    cfg.num_updates, cfg.total_num_steps,
                )
            )
        if cfg.num_updates == -1 and cfg.total_num_steps == -1:
            raise RuntimeError(
                "One of `num_updates` and `total_num_steps` must be specified.\n"
                "`num_updates`: {} `total_num_steps`: {}".format(
                    cfg.num_updates, cfg.total_num_steps,
                )
            )
        if cfg.num_ckpts != -1 and cfg.ckpt_interval != -1:
            raise RuntimeError(
                "`num_ckpts` and `ckpt_interval` are both specified."
                "  One must be -1.\n"
                " `num_ckpts`: {} `ckpt_interval`: {}".format(
                    cfg.num_ckpts, cfg.ckpt_interval
                )
            )

        if cfg.num_ckpts == -1 and cfg.ckpt_interval == -1:
            raise RuntimeError(
                "One of `num_ckpts` and `ckpt_interval` must be specified"
                " `num_ckpts`: {} `ckpt_interval`: {}".format(
                    cfg.num_ckpts, cfg.ckpt_interval
                )
            )

    def percent_done(self) -> float:
        if self.cfg.num_updates != -1:
            return self.num_updates_done / self.cfg.num_updates
        else:
            return self.num_steps_done / self.cfg.total_num_steps

    def is_done(self) -> bool:
        return self.percent_done() >= 1.0

    def should_checkpoint(self) -> bool:
        needs_checkpoint = False
        if self.cfg.num_ckpts != -1:
            checkpoint_every = 1 / self.cfg.num_ckpts
            if (
                self._last_checkpoint_percent + checkpoint_every
                < self.percent_done()
            ):
                needs_checkpoint = True
                self._last_checkpoint_percent = self.percent_done()
        else:
            needs_checkpoint = (
                self.num_updates_done % self.cfg.ckpt_interval
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
        ckpt_dir = self.cfg.ckpt_dir.format(
            results_root=self.cfg.results_root,
            run_id=str(self.cfg.run_id),
        )
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        return ckpt_dir

    @property
    def tb_dir(self) -> PathLike:
        tb_dir = self.cfg.tb_dir.format(
            tb_root=self.cfg.tb_root,
            run_id=str(self.cfg.run_id),
        )
        if not os.path.exists(tb_dir):
            os.makedirs(tb_dir, exist_ok=True)
        return tb_dir

    @property
    def video_dir(self) -> PathLike:
        video_dir = self.cfg.video_dir.format(
            results_root=self.cfg.results_root,
            run_id=self.cfg.run_id,
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

    @staticmethod
    def _pause_envs(
        envs_to_pause: List[int],
        envs: VecEnv,
        test_recurrent_hidden_states: torch.Tensor,
        not_done_masks: torch.Tensor,
        current_episode_reward: torch.Tensor,
        prev_actions: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        rgb_frames: Union[List[List[Any]], List[List[np.ndarray]]],
    ) -> Tuple[
        VecEnv,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Dict[str, torch.Tensor],
        List[List[Any]],
    ]:
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                state_index
            ]
            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            test_recurrent_hidden_states,
            not_done_masks,
            current_episode_reward,
            prev_actions,
            batch,
            rgb_frames,
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
