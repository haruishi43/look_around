#!/usr/bin/env python3

import argparse
import os
import random
from typing import Any, Dict, Optional

import torch
from gym.spaces import Box
from gym.spaces import Dict as SpaceDict
from gym.spaces import Discrete

from LookAround.config import Config
from LookAround.core.agent import Agent

from findview_baselines.rl.ppo.policy import FindViewBaselinePolicy


class PPOAgent(Agent):
    def __init__(
        self,
        cfg: Config,
        ckpt_path: os.PathLike = None,
    ) -> None:
        observation_space = SpaceDict(
            {
                "pers": Box(
                    low=torch.finfo(torch.float32).min,
                    high=torch.finfo(torch.float32).max,
                    shape=(
                        3,
                        cfg.sim.height,
                        cfg.sim.width,
                    ),
                ),
                "target": Box(
                    low=torch.finfo(torch.float32).min,
                    high=torch.finfo(torch.float32).max,
                    shape=(
                        3,
                        cfg.sim.height,
                        cfg.sim.width,
                    ),
                ),
            }
        )
        action_spaces = Discrete(5)

        test_cfg = cfg.test

        self.device = (
            torch.device("cuda:{}".format(test_cfg.device))
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        ppo_cfg = cfg.ppo
        self.hidden_size = ppo_cfg.hidden_size

        random.seed(cfg.seed)
        torch.random.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True  # type: ignore

        self.actor_critic = FindViewBaselinePolicy(
            observation_space=observation_space,
            action_space=action_spaces,
            hidden_size=self.hidden_size,
            **cfg.policy,
        )
        self.actor_critic.to(self.device)

        ckpt = torch.load(ckpt_path, map_location=self.device)
        #  Filter only actor_critic weights
        self.actor_critic.load_state_dict(
            {
                k[len("actor_critic.") :]: v
                for k, v in ckpt["state_dict"].items()
                if "actor_critic" in k
            }
        )

        self.test_recurrent_hidden_states: Optional[torch.Tensor] = None
        self.not_done_masks: Optional[torch.Tensor] = None
        self.prev_actions: Optional[torch.Tensor] = None

    @classmethod
    def from_config(cls, cfg: Config, ckpt_filename: str):
        # FIXME: sort out parameters
        # configs should be outside `__init__`
        return cls(cfg=cfg, ckpt_filename=ckpt_filename)

    def reset(self) -> None:
        self.test_recurrent_hidden_states = torch.zeros(
            1,
            self.actor_critic.net.num_recurrent_layers,
            self.hidden_size,
            device=self.device,
        )
        self.not_done_masks = torch.zeros(
            1, 1, device=self.device, dtype=torch.bool
        )
        self.prev_actions = torch.zeros(
            1, 1, dtype=torch.long, device=self.device
        )

    def act(self, observations: Dict[str, Any]) -> Dict[str, int]:

        # NOTE: need to extend observation to 4 dims if not 4 dim
        for name, value in observations.items():
            if torch.is_tensor(value):
                if len(value.shape) == 3:
                    observations[name] = value.unsqueeze(0)

        with torch.no_grad():
            (
                _,
                actions,
                _,
                self.test_recurrent_hidden_states,
            ) = self.actor_critic.act(
                observations,
                self.test_recurrent_hidden_states,
                self.prev_actions,
                self.not_done_masks,
                deterministic=False,
            )
            #  Make masks not done till reset (end of episode) will be called
            self.not_done_masks.fill_(True)
            self.prev_actions.copy_(actions)  # type: ignore

        return {"action": actions[0][0].item()}


def main():

    from LookAround.core.logging import logger
    from LookAround.FindView.benchmark import FindViewBenchmark

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
    )
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)

    ckpt_path = args.ckpt_path

    if not os.path.exists(ckpt_path):
        ckpt_dir = cfg.trainer.ckpt_dir.format(
            results_root=cfg.results_root,
            run_id=cfg.trainer.run_id,
        )
        ckpt_path =

        if ckpt_filename is None:
            print("warning: using ckpt from config file")
            ckpt_path = os.path.join(
                ckpt_path,
                test_cfg.ckpt_path,
            )
        else:
            ckpt_path = os.path.join(
                ckpt_path,
                ckpt_filename,
            )
            print(f"loading from {ckpt_path}")

    assert os.path.exists(ckpt_path), \
        f"{ckpt_path} doesn't exist!"

    agent = PPOAgent(cfg, ckpt_filename=args.ckpt_filename)
    benchmark = FindViewBenchmark(
        cfg=cfg,
        device=agent.device,
    )
    if args.num_episodes == 0:
        metrics = benchmark.evaluate(agent, num_episodes=None)
    else:
        metrics = benchmark.evaluate(agent, num_episodes=args.num_episodes)
    for k, v in metrics.items():
        logger.info("{}: {:.3f}".format(k, v))


if __name__ == "__main__":
    main()
