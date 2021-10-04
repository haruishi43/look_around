#!/usr/bin/env python3

from collections import defaultdict
import os
from os import PathLike
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn

from LookAround.config import Config
from LookAround.core import logger
from LookAround.FindView.vec_env import VecEnv

from findview_baselines.common.tensorboard_utils import TensorboardWriter
from findview_baselines.utils.common import (
    get_checkpoint_id,
    poll_checkpoint_folder,
)


class BaseValidator(object):

    # Properties
    cfg: Config

    # Hidden Properties
    _flush_secs: int

    def __init__(
        self,
        cfg: Config,
    ) -> None:

        # Initialize properties
        self.cfg = cfg

        # Initialize hidden properties
        self._flush_secs = 30

    @property
    def flush_secs(self) -> int:
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

    def __call__(self, agent: nn.Module):
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
            self._base_cfg.log_file.format(
                split="test",
                log_root=self.cfg.log_root,
                run_id=self._base_cfg.run_id,
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


class BaseRLValidator(BaseValidator):

    # Properties
    device: torch.device
    video_option: List[str]

    def __init__(self, cfg: Config) -> None:
        super().__init__()

        # initialize properties
        self.cfg = cfg
        self.video_option = self.cfg.base_trainer.video_option

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
        # pausing envs with no new episode
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
