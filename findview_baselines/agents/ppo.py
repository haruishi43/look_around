#!/usr/bin/env python3

# FIXME: somehow, running this script calls a warning:
# /home/ubuntu/.pyenv/versions/3.8.8/lib/python3.8/site-packages/setuptools/distutils_patch.py:25: User
# Warning: Distutils was imported before Setuptools. This usage is discouraged and may exhibit undesira
# ble behaviors or errors. Please use Setuptools' objects directly or at least import Setuptools first.

import os
import random
from typing import Any, Dict, Optional

import torch
from gymnasium.spaces import (
    Box,
    Dict as SpaceDict,
    Discrete,
)

from LookAround.config import Config
from LookAround.core.agent import Agent

from findview_baselines.rl.models.base_policy import FindViewBaselinePolicy


class PPOAgent(Agent):
    def __init__(
        self,
        cfg: Config,
        ckpt_path: os.PathLike,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device(0),
    ) -> None:
        assert os.path.exists(ckpt_path), f"ERR: {ckpt_path} doesn't exist!"

        if torch.cuda.is_available():
            self.device = torch.device(device)
        else:
            self.device = torch.device("cpu")

        observation_space = SpaceDict(
            {
                "pers": Box(
                    low=torch.finfo(dtype).min,
                    high=torch.finfo(dtype).max,
                    shape=(
                        3,
                        cfg.sim.height,
                        cfg.sim.width,
                    ),
                ),
                "target": Box(
                    low=torch.finfo(dtype).min,
                    high=torch.finfo(dtype).max,
                    shape=(
                        3,
                        cfg.sim.height,
                        cfg.sim.width,
                    ),
                ),
            }
        )
        action_spaces = Discrete(5)

        random.seed(cfg.seed)
        torch.random.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True  # type: ignore

        # loading values from ckpt dict
        ckpt_dict = torch.load(ckpt_path, map_location=self.device)

        agent_name = "ppo"
        ckpt_cfg = ckpt_dict["cfg"]

        # try to get trainer
        trainer = ckpt_cfg.get("trainer", None)
        if trainer is None:
            # backward compatibility
            trainer = ckpt_cfg.get("base_trainer", None)
            assert trainer is not None, ckpt_cfg.pretty_text
            agent_name += f"_compat_{str(trainer.run_id)}"
        else:
            agent_name += f"_{str(trainer.run_id)}"

        # add rl_env name
        agent_name += f"_{ckpt_cfg.rl_env.name}"

        # add identifier
        identifier = trainer.get("identifier", None)
        if identifier is not None:
            agent_name += f"_{identifier}"

        step_id = ckpt_dict["extra_state"]["num_steps_done"]
        agent_name += f"_{str(step_id)}"

        self.name = agent_name  # NOTE: setting agent name here!

        # load model configuration from ckpt
        ppo_cfg = ckpt_cfg.ppo
        self.hidden_size = ppo_cfg.hidden_size

        self.actor_critic = FindViewBaselinePolicy(
            observation_space=observation_space,
            action_space=action_spaces,
            hidden_size=self.hidden_size,
            **ckpt_cfg.policy,
        )
        self.actor_critic.to(self.device)

        #  Filter only actor_critic weights
        self.actor_critic.load_state_dict(
            {
                k[len("actor_critic.") :]: v
                for k, v in ckpt_dict["state_dict"].items()
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
    import argparse

    from LookAround.config import DictAction
    from LookAround.FindView.benchmark import FindViewBenchmark
    from LookAround.FindView.benchmark import CorruptedFindViewBenchmark
    from LookAround.FindView.corruptions import get_corruption_names

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--name",
        type=str,
        help="name of the agent (used for naming save directory)",
    )
    parser.add_argument(
        "--corrupted",
        action="store_true",
        help="use corrupted",
    )
    parser.add_argument(
        "--all", action="store_true", help="benchmark all corruptions"
    )
    parser.add_argument(
        "--options",
        nargs="+",
        action=DictAction,
        help="arguments in dict",
    )
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    print(">>> Config:")
    print(cfg.pretty_text)

    # Intializing the agent
    ckpt_path = args.ckpt_path
    assert os.path.exists(ckpt_path), f"ERR: {ckpt_path} doesn't exist!"

    if torch.cuda.is_available():
        device = torch.device(cfg.benchmark.device)
    else:
        device = torch.device("cpu")

    agent = PPOAgent(
        cfg=cfg,
        ckpt_path=ckpt_path,
        device=device,
    )

    name = agent.name
    if args.name is not None:
        name += "_" + args.name

    # Benchmark
    print(f"Benchmarking {name}")
    if args.corrupted:
        num_episodes = 60
        # TODO: create a script that evaluates each corruptions
        if args.all:
            corruptions = get_corruption_names("all")
            for corruption in corruptions:
                cfg.benchmark.corruption = corruption
                benchmark = CorruptedFindViewBenchmark(
                    cfg=cfg,
                    agent_name=name,
                )
                benchmark.evaluate(agent, num_episodes)
        else:
            benchmark = CorruptedFindViewBenchmark(
                cfg=cfg,
                agent_name=name,
            )
            benchmark.evaluate(agent, num_episodes)
    else:
        benchmark = FindViewBenchmark(
            cfg=cfg,
            agent_name=name,
        )
        benchmark.evaluate(agent)


if __name__ == "__main__":
    main()
