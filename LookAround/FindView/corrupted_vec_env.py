#!/usr/bin/env python3

"""Naive Vectorized Environment for RL

Reference:
https://github.com/facebookresearch/habitat-lab/blob/master/habitat/core/vector_env.py

"""

from copy import deepcopy
from functools import partial
import random
from multiprocessing.connection import Connection
from multiprocessing.context import BaseContext
from queue import Queue
from threading import Thread
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
    cast,
)
import warnings

import attr
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
from torch import multiprocessing as mp  # type:ignore

from LookAround.config import Config
from LookAround.core.logging import logger
from LookAround.FindView.dataset import Episode, PseudoEpisode, make_dataset
from LookAround.FindView.corrupted_env import CorruptedFindViewEnv
from LookAround.FindView.corrupted_rl_env import CorruptedFindViewRLEnv, CorruptedRLEnvRegistry
from LookAround.FindView.sim import deg2rad
from LookAround.utils.visualizations import tile_images
from LookAround.utils.pickle5_multiprocessing import ConnectionWrapper


STEP_COMMAND = "step"
RESET_COMMAND = "reset"
RENDER_COMMAND = "render"
CLOSE_COMMAND = "close"
CALL_COMMAND = "call"
CHANGE_DIFFICULTY_COMMAND = "change_difficulty"
CHANGE_SEVERITY_COMMAND = "change_severity"

EPISODE_OVER_NAME = "episode_over"
GET_METRICS_NAME = "get_metrics"
CURRENT_EPISODE_NAME = "current_episode"
NUMBER_OF_EPISODE_NAME = "number_of_episodes"
ACTION_SPACE_NAME = "action_space"
OBSERVATION_SPACE_NAME = "observation_space"


@attr.s(auto_attribs=True, slots=True)
class _ReadWrapper:
    """Convenience wrapper to track if a connection to a worker process
    should have something to read.
    """
    read_fn: Callable[[], Any]
    rank: int
    is_waiting: bool = False

    def __call__(self) -> Any:
        if not self.is_waiting:
            raise RuntimeError(
                f"Tried to read from process {self.rank}"
                " but there is nothing waiting to be read"
            )
        res = self.read_fn()
        self.is_waiting = False

        return res


@attr.s(auto_attribs=True, slots=True)
class _WriteWrapper:
    """Convenience wrapper to track if a connection to a worker process
    can be written to safely.  In other words, checks to make sure the
    result returned from the last write was read.
    """
    write_fn: Callable[[Any], None]
    read_wrapper: _ReadWrapper

    def __call__(self, data: Any) -> None:
        if self.read_wrapper.is_waiting:
            raise RuntimeError(
                f"Tried to write to process {self.read_wrapper.rank}"
                " but the last write has not been read"
            )
        self.write_fn(data)
        self.read_wrapper.is_waiting = True


class CorruptedVecEnv(object):

    # Properties
    action_spaces: List[spaces.Dict]
    observation_spaces: List[spaces.Dict]
    number_of_episodes: Optional[int]

    # Hidden Properties
    _num_envs: int
    _auto_reset_done: bool

    @property
    def num_envs(self) -> int:
        raise NotImplementedError

    def current_episodes(self) -> List[Episode]:
        raise NotImplementedError

    def episode_over(self) -> List[bool]:
        raise NotImplementedError

    def get_metrics(self) -> List[Any]:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def reset_at(self, index: int):
        raise NotImplementedError

    def step(self, actions) -> List[Any]:
        raise NotImplementedError

    def step_at(self, index: int, action):
        raise NotImplementedError

    def async_step_at(self, index: int, action) -> None:
        raise NotImplementedError

    def wait_step_at(self, index: int) -> List[Any]:
        raise NotImplementedError

    def pause_at(self, index: int):
        raise NotImplementedError

    def resume_all(self):
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def change_difficulty(self, difficulty: str, bounded: bool) -> None:
        raise NotImplementedError

    def change_severity(self, severity: int) -> None:
        raise NotImplementedError


class CorruptedMPVecEnv(CorruptedVecEnv):

    # Hidden Properties
    _workers: List[Union[mp.Process, Thread]]
    _mp_ctx: BaseContext
    _connection_read_fns: List[_ReadWrapper]
    _connection_write_fns: List[_WriteWrapper]

    def __init__(
        self,
        make_env_fn,
        env_fn_kwargs,
        auto_reset_done: bool = True,
        multiprocessing_start_method: str = "forkserver",
    ) -> None:

        self._num_envs = len(env_fn_kwargs)

        assert multiprocessing_start_method in self._valid_start_methods, (
            "multiprocessing_start_method must be one of {}. Got '{}'"
        ).format(self._valid_start_methods, multiprocessing_start_method)
        self._auto_reset_done = auto_reset_done
        self._mp_ctx = mp.get_context(multiprocessing_start_method)
        self._workers = []
        (
            self._connection_read_fns,
            self._connection_write_fns,
        ) = self._spawn_workers(  # noqa
            env_fn_kwargs,
            make_env_fn,
        )

        self._is_closed = False

        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (OBSERVATION_SPACE_NAME, None)))
        self.observation_spaces = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (ACTION_SPACE_NAME, None)))
        self.action_spaces = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (NUMBER_OF_EPISODE_NAME, None)))
        self.number_of_episodes = [
            read_fn() for read_fn in self._connection_read_fns
        ]
        self._paused: List[Tuple] = []

    @property
    def num_envs(self):
        """number of individual environments."""
        return self._num_envs - len(self._paused)

    @staticmethod
    def _worker_env(
        connection_read_fn: Callable,
        connection_write_fn: Callable,
        env_fn: Callable,
        env_fn_kwargs: Tuple[Any],
        auto_reset_done: bool,
        child_pipe: Optional[Connection] = None,
        parent_pipe: Optional[Connection] = None,
    ) -> None:
        """process worker for creating and interacting with the environment."""

        env = env_fn(**env_fn_kwargs)
        if parent_pipe is not None:
            parent_pipe.close()
        try:
            command, data = connection_read_fn()
            while command != CLOSE_COMMAND:
                if command == STEP_COMMAND:
                    # different step methods for habitat.RLEnv and habitat.Env
                    if isinstance(env, (CorruptedFindViewRLEnv, gym.Env)):
                        observations, reward, done, info = env.step(**data)
                        if auto_reset_done and done:
                            # FIXME: last observation is not taken into account during validation
                            observations = env.reset()
                        connection_write_fn(
                            (observations, reward, done, info)
                        )
                    elif isinstance(env, CorruptedFindViewEnv):  # type: ignore
                        observations = env.step(**data)
                        done = env.episode_over
                        metrics = env.get_metrics()
                        # NOTE: don't restart since `episode_over` also gets reset
                        if auto_reset_done and done:
                            observations = env.reset()
                        connection_write_fn((observations, done, metrics))
                    else:
                        raise NotImplementedError

                elif command == RESET_COMMAND:
                    observations = env.reset()
                    connection_write_fn(observations)

                elif command == RENDER_COMMAND:
                    connection_write_fn(env.render(*data[0], **data[1]))

                elif command == CHANGE_DIFFICULTY_COMMAND:
                    connection_write_fn(env.change_difficulty(**data))

                elif command == CHANGE_SEVERITY_COMMAND:
                    connection_write_fn(env.change_severity(**data))

                elif command == CALL_COMMAND:
                    function_name, function_args = data
                    if function_args is None:
                        function_args = {}

                    result_or_fn = getattr(env, function_name)

                    if len(function_args) > 0 or callable(result_or_fn):
                        result = result_or_fn(**function_args)
                    else:
                        result = result_or_fn

                    connection_write_fn(result)

                else:
                    raise NotImplementedError(f"Unknown command {command}")

                command, data = connection_read_fn()

        except KeyboardInterrupt:
            logger.info("Worker KeyboardInterrupt")
        finally:
            if child_pipe is not None:
                child_pipe.close()
            env.close()

    def _spawn_workers(
        self,
        env_fn_kwargs: Sequence[Dict],
        make_env_fn: Callable[..., Union[CorruptedFindViewEnv, CorruptedFindViewRLEnv]],
    ) -> Tuple[List[_ReadWrapper], List[_WriteWrapper]]:
        parent_connections, worker_connections = zip(
            *[
                [ConnectionWrapper(c) for c in self._mp_ctx.Pipe(duplex=True)]
                for _ in range(self._num_envs)
            ]
        )
        self._workers = []
        for worker_conn, parent_conn, env_kwargs in zip(
            worker_connections, parent_connections, env_fn_kwargs
        ):
            ps = self._mp_ctx.Process(
                target=self._worker_env,
                args=(
                    worker_conn.recv,
                    worker_conn.send,
                    make_env_fn,
                    env_kwargs,
                    self._auto_reset_done,
                    worker_conn,
                    parent_conn,
                ),
            )
            self._workers.append(cast(mp.Process, ps))
            ps.daemon = True
            ps.start()
            worker_conn.close()

        read_fns = [
            _ReadWrapper(p.recv, rank)
            for rank, p in enumerate(parent_connections)
        ]
        write_fns = [
            _WriteWrapper(p.send, read_fn)
            for p, read_fn in zip(parent_connections, read_fns)
        ]

        return read_fns, write_fns

    def current_episodes(self):
        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (CURRENT_EPISODE_NAME, None)))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        return results

    def episode_over(self):
        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (EPISODE_OVER_NAME, None)))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        return results

    def get_metrics(self):
        for write_fn in self._connection_write_fns:
            write_fn((CALL_COMMAND, (GET_METRICS_NAME, None)))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        return results

    def reset(self):
        """Reset all the vectorized environments
        :return: list of outputs from the reset method of envs.
        """
        for write_fn in self._connection_write_fns:
            write_fn((RESET_COMMAND, None))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        return results

    def reset_at(self, index: int):
        """Reset in the index_env environment in the vector.
        :param index_env: index of the environment to be reset
        :return: list containing the output of reset method of indexed env.
        """
        self._connection_write_fns[index]((RESET_COMMAND, None))
        return self._connection_read_fns[index]()

    def async_step_at(
        self,
        index: int,
        action: Union[int, str, Dict[str, Any]],
    ) -> None:
        # Backward compatibility
        if isinstance(action, (int, np.integer, str)):
            action = {"action": {"action": action}}

        self._warn_cuda_tensors(action)
        self._connection_write_fns[index]((STEP_COMMAND, action))

    def wait_step_at(self, index: int) -> Any:
        return self._connection_read_fns[index]()

    def step_at(self, index: int, action: Union[int, str, Dict[str, Any]]):
        """Step in the index environment in the vector.
        :param index: index of the environment to be stepped into
        :param action: action to be taken
        :return: list containing the output of step method of indexed env.
        """
        self.async_step_at(index, action)
        return self.wait_step_at(index)

    def async_step(
        self,
        actions: Sequence[Union[int, str, Dict[str, Any]]],
    ) -> None:
        """Asynchronously step in the environments.
        :param actions: list of size _num_envs containing keyword arguments to
            pass to `step` method for each Environment. For example,
            `[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        """

        for index, action in enumerate(actions):
            self.async_step_at(index, action)

    def wait_step(self) -> List[Any]:
        """Wait until all the asynchronized environments have synchronized."""
        return [
            self.wait_step_at(i) for i in range(self.num_envs)
        ]

    def step(
        self,
        actions: Sequence[Union[int, str, Dict[str, Any]]],
    ) -> List[Any]:
        """Perform actions in the vectorized environments.
        :param actions: list of size _num_envs containing keyword arguments to
            pass to `step` method for each Environment. For example,
            `[{"action": "TURN_LEFT", "action_args": {...}}, ...]`.
        :return: list of outputs from the step method of envs.
        """
        self.async_step(actions)
        return self.wait_step()

    def close(self) -> None:
        if self._is_closed:
            return

        for read_fn in self._connection_read_fns:
            if read_fn.is_waiting:
                read_fn()

        for write_fn in self._connection_write_fns:
            write_fn((CLOSE_COMMAND, None))

        for _, _, write_fn, _ in self._paused:
            write_fn((CLOSE_COMMAND, None))

        for process in self._workers:
            process.join()

        for _, _, _, process in self._paused:
            process.join()

        self._is_closed = True

    def pause_at(self, index: int) -> None:
        """Pauses computation on this env without destroying the env.
        :param index: which env to pause. All indexes after this one will be
            shifted down by one.
        This is useful for not needing to call steps on all environments when
        only some are active (for example during the last episodes of running
        eval episodes).
        """
        if self._connection_read_fns[index].is_waiting:
            self._connection_read_fns[index]()
        read_fn = self._connection_read_fns.pop(index)
        write_fn = self._connection_write_fns.pop(index)
        worker = self._workers.pop(index)
        self._paused.append((index, read_fn, write_fn, worker))

    def resume_all(self) -> None:
        """Resumes any paused envs."""
        for index, read_fn, write_fn, worker in reversed(self._paused):
            self._connection_read_fns.insert(index, read_fn)
            self._connection_write_fns.insert(index, write_fn)
            self._workers.insert(index, worker)
        self._paused = []

    def call_at(
        self,
        index: int,
        function_name: str,
        function_args: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """Calls a function or retrieves a property/member variable (which is passed by name)
        on the selected env and returns the result.
        :param index: which env to call the function on.
        :param function_name: the name of the function to call or property to retrieve on the env.
        :param function_args: optional function args.
        :return: result of calling the function.
        """
        self._connection_write_fns[index](
            (CALL_COMMAND, (function_name, function_args))
        )
        result = self._connection_read_fns[index]()
        return result

    def call(
        self,
        function_names: List[str],
        function_args_list: Optional[List[Any]] = None,
    ) -> List[Any]:
        """Calls a list of functions (which are passed by name) on the
        corresponding env (by index).
        :param function_names: the name of the functions to call on the envs.
        :param function_args_list: list of function args for each function. If
            provided, :py:`len(function_args_list)` should be as long as
            :py:`len(function_names)`.
        :return: result of calling the function.
        """
        if function_args_list is None:
            function_args_list = [None] * len(function_names)
        assert len(function_names) == len(function_args_list)
        func_args = zip(function_names, function_args_list)
        for write_fn, func_args_on in zip(
            self._connection_write_fns, func_args
        ):
            write_fn((CALL_COMMAND, func_args_on))
        results = []
        for read_fn in self._connection_read_fns:
            results.append(read_fn())
        return results

    def render(
        self, *args, **kwargs,
    ) -> Union[Dict[str, np.ndarray], None]:
        """Render observations from all environments in a tiled image."""
        for write_fn in self._connection_write_fns:
            write_fn((RENDER_COMMAND, (args, kwargs)))
        renders = [read_fn() for read_fn in self._connection_read_fns]
        pers = [r['pers'] for r in renders]
        target = [r['target'] for r in renders]
        pers = tile_images(pers)
        target = tile_images(target)
        return {
            "pers": pers,
            "target": target
        }

    def change_difficulty(
        self,
        difficulty: str,
        bounded: bool,
    ) -> None:
        for write_fn in self._connection_write_fns:
            write_fn(
                (
                    CHANGE_DIFFICULTY_COMMAND,
                    dict(difficulty=difficulty, bounded=bounded)
                )
            )
        # NOTE: need to read before another write
        _ = [read_fn() for read_fn in self._connection_read_fns]

    def change_severity(
        self,
        severity: int,
    ) -> None:
        for write_fn in self._connection_write_fns:
            write_fn(
                (
                    CHANGE_SEVERITY_COMMAND,
                    dict(severity=severity)
                )
            )
        # NOTE: need to read before another write
        _ = [read_fn() for read_fn in self._connection_read_fns]

    @property
    def _valid_start_methods(self) -> Set[str]:
        return {"forkserver", "spawn", "fork"}

    def _warn_cuda_tensors(
        self, action: Dict[str, Any], prefix: Optional[str] = None
    ):
        if torch is None:
            return

        for k, v in action.items():
            if isinstance(v, dict):
                subk = f"{prefix}.{k}" if prefix is not None else k
                self._warn_cuda_tensors(v, prefix=subk)
            elif torch.is_tensor(v) and v.device.type == "cuda":
                subk = f"{prefix}.{k}" if prefix is not None else k
                warnings.warn(
                    "Action with key {} is a CUDA tensor."
                    "  This will result in a CUDA context in the subproccess worker."
                    "  Using CPU tensors instead is recommended.".format(subk)
                )


class CorruptedThreadedVecEnv(CorruptedMPVecEnv):
    """Provides same functionality as `VecEnv`, the only difference
    is it runs in a multi-thread setup inside a single process.
    The `VecEnv` runs in a multi-proc setup. This makes it much easier
    to debug when using `VecEnv` because you can actually put break
    points in the environment methods. It should not be used for best
    performance.
    """

    def _spawn_workers(
        self,
        env_fn_kwargs: Sequence[Dict],
        make_env_fn: Callable[..., Union[CorruptedFindViewEnv, CorruptedFindViewRLEnv]],
    ) -> Tuple[List[_ReadWrapper], List[_WriteWrapper]]:
        queues: Iterator[Tuple[Any, ...]] = zip(
            *[(Queue(), Queue()) for _ in range(self._num_envs)]
        )
        parent_read_queues, parent_write_queues = queues
        self._workers = []
        for parent_read_queue, parent_write_queue, env_kwargs in zip(
            parent_read_queues, parent_write_queues, env_fn_kwargs
        ):
            thread = Thread(
                target=self._worker_env,
                args=(
                    parent_write_queue.get,
                    parent_read_queue.put,
                    make_env_fn,
                    env_kwargs,
                    self._auto_reset_done,
                    None,  # child_conn
                    None,  # parent_conn
                ),
            )
            self._workers.append(thread)
            thread.daemon = True
            thread.start()

        read_fns = [
            _ReadWrapper(q.get, rank)
            for rank, q in enumerate(parent_read_queues)
        ]
        write_fns = [
            _WriteWrapper(q.put, read_wrapper)
            for q, read_wrapper in zip(parent_write_queues, read_fns)
        ]
        return read_fns, write_fns


class CorruptedEquilibVecEnv(CorruptedVecEnv):

    # Properties
    envs: List[Union[CorruptedFindViewEnv, CorruptedFindViewEnv]]

    def __init__(
        self,
        make_env_fn,
        env_fn_kwargs,
        auto_reset_done: bool = True,
    ) -> None:

        self._num_envs = len(env_fn_kwargs)
        self._auto_reset_done = auto_reset_done

        # initialize envs
        self.envs = [
            make_env_fn(**env_fn_kwargs[i])
            for i in range(self._num_envs)
        ]

        self.action_spaces = [env.action_space for env in self.envs]
        self.observation_spaces = [env.observation_space for env in self.envs]
        self.number_of_episodes = [env.number_of_episodes for env in self.envs]
        self._paused: List[Tuple] = []

    @property
    def num_envs(self):
        """number of individual environments.
        """
        return self._num_envs - len(self._paused)

    def current_episodes(self):
        results = []
        for env in self.envs:
            results.append(env.current_episode)
        return results

    def reset(self):
        batch_obs = []
        for env in self.envs:
            batch_obs.append(env.reset())
        return batch_obs

    def reset_at(self, index: int):
        obs = self.envs[index].reset()
        return obs

    def step(self, actions: List[str]):
        batch_ret = []

        # Batched method
        # FIXME: still slow... I don't know how to fix it...
        # get rotations
        rots = []
        for env, action in zip(self.envs, actions):
            rot = env.step_before(action)
            rots.append(rot)

        # NOTE: really hacky way of batch sampling
        sims = [env.sim for env in self.envs]

        is_torch = sims[0].is_torch
        none_idx = [i for i, rot in enumerate(rots) if rot is None]

        rad_rots = [deg2rad(rot) for rot in rots if rot is not None]

        batched_equi = []
        for i, sim in enumerate(sims):
            if i not in none_idx:
                batched_equi.append(sim.equi)

        if len(batched_equi) == 0:
            return

        if is_torch:
            if len(batched_equi) == 1:
                batched_equi = batched_equi[0].unsqueeze(0)
            else:
                batched_equi = torch.stack(batched_equi, dim=0)
        else:
            if len(batched_equi) == 1:
                batched_equi = batched_equi[0][None, ...]
            else:
                batched_equi = np.stack(batched_equi, axis=0)

        assert len(batched_equi.shape) == 4
        batched_pers = sims[0].equi2pers(batched_equi, rots=rad_rots)
        assert len(batched_pers.shape) == 4

        count = 0
        for i, sim in enumerate(sims):
            # if `rot` was None, `sim` should keep the original `pers`
            if i not in none_idx:
                sim._pers = batched_pers[count]
                count += 1

        assert count == len(batched_pers)

        # make sure to get observations
        for env in self.envs:
            if isinstance(env, CorruptedFindViewRLEnv):
                observation, reward, done, info = env.step_after()

                if done and self._auto_reset_done:
                    observation = env.reset()

                batch_ret.append(
                    (observation, reward, done, info)
                )
            elif isinstance(env, CorruptedFindViewEnv):
                observation = env.step_after()
                done = env.episode_over
                metrics = env.get_metrics()
                if done and self._auto_reset_done:
                    observation = env.reset()
                batch_ret.append((observation, done, metrics))

        # Serial method
        # NOTE: pretty slow
        # for i, env in enumerate(self.envs):
        #     observation, reward, done, info = env.step(actions[i])

        #     if done:
        #         observation = env.reset()

        #     batch_ret.append(
        #         (observation, reward, done, info)
        #     )

        return batch_ret

    def async_step_at(self, index, action) -> None:
        raise NotImplementedError

    def wait_step_at(self, index) -> List[Any]:
        raise NotImplementedError

    def step_at(self, index: int, action: str):
        return self.envs[index].step(action)

    def pause_at(self, index: int) -> None:
        env = self.envs.pop(index)
        self._paused.append((index, env))

    def resume_all(self) -> None:
        for index, env in reversed(self._paused):
            self.envs.insert(index, env)
        self._paused = []

    def render(self):
        renders = [env.render() for env in self.envs]
        pers = [r['pers'] for r in renders]
        target = [r['target'] for r in renders]
        pers_tile = tile_images(pers)
        target_tile = tile_images(target)
        return {
            'pers': pers_tile,
            'target': target_tile,
        }

    def change_difficulty(self, difficulty: str, bounded: bool) -> None:
        for env in self.envs:
            env.change_difficulty(difficulty=difficulty, bounded=bounded)

    def change_severity(self, severity: int) -> None:
        for env in self.envs:
            env.change_severity(severity=severity)

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# filters for making datasets

def filter_by_name(
    episode: Union[Episode, PseudoEpisode],
    names: List[str],
) -> bool:
    return episode.img_name in names


def filter_by_sub_labels(
    episode: Union[Episode, PseudoEpisode],
    sub_labels: List[str],
) -> bool:
    return episode.sub_label in sub_labels


def filter_by_difficulty(
    episode: Episode,
    difficulties: List[str],
) -> bool:
    return episode.difficulty in difficulties


# corrupted env/rlenv initialization function

def make_corrupted_env_fn(
    cfg: Config,
    filter_fn: Callable[..., bool],
    split: str,
    rank: int,
    num_episodes_per_img: int = -1,
    dtype: Union[np.dtype, torch.dtype] = torch.float32,
    device: torch.device = torch.device('cpu'),
) -> CorruptedFindViewEnv:

    # FIXME: play around with this value for threaded and multiprocessing
    # Lower threads the better
    torch.set_num_threads(1)  # NOTE: this is needed for multiprocessing?
    env: CorruptedFindViewEnv = CorruptedFindViewEnv.from_config(
        cfg=cfg,
        split=split,
        filter_fn=filter_fn,
        num_episodes_per_img=num_episodes_per_img,
        dtype=dtype,
        device=device,
    )
    env.seed(rank)
    return env


def make_corrupted_rl_env_fn(
    cfg: Config,
    filter_fn: Callable[..., bool],
    split: str,
    rank: int,
    num_episodes_per_img: int = -1,
    dtype: Union[np.dtype, torch.dtype] = torch.float32,
    device: torch.device = torch.device('cpu'),
) -> CorruptedFindViewRLEnv:

    # FIXME: play around with this value for threaded and multiprocessing
    # Lower threads the better
    torch.set_num_threads(1)  # NOTE: this is needed for multiprocessing?
    rlenv: CorruptedFindViewRLEnv = CorruptedRLEnvRegistry.build(
        cfg,
        name=cfg.rl_env.name,
        filter_fn=filter_fn,
        num_episodes_per_img=num_episodes_per_img,
        split=split,
        dtype=dtype,
        device=device,
    )
    rlenv.seed(rank)
    return rlenv


# function to construct vectorized envs

def construct_corrupted_envs(
    cfg: Config,
    split: str,
    is_rlenv: bool = True,
    dtype: Union[np.dtype, torch.dtype] = torch.float32,
    device: torch.device = torch.device('cpu'),
    vec_type: str = "threaded",
    auto_reset_done: bool = True,
) -> CorruptedVecEnv:
    """Basic initialization of vectorized environments

    It splits the dataset into smaller dataset for each enviornment by first
    splitting the dataset by `img_name`.
    """

    # 1. preprocessing
    # NOTE: make sure to seed first so that we get consistant tests
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    num_envs = cfg.trainer.num_envs

    # get all dataset
    dataset = make_dataset(cfg=cfg, split=split)
    img_names = dataset.get_img_names()

    if len(img_names) > 0:
        # NOTE: uses global random state
        random.shuffle(img_names)

        assert len(img_names) >= num_envs, (
            "reduce the number of environments as there "
            "aren't enough diversity in images"
        )

    img_name_splits = [[] for _ in range(num_envs)]
    for idx, img_name in enumerate(img_names):
        img_name_splits[idx % len(img_name_splits)].append(img_name)

    assert sum(map(len, img_name_splits)) == len(img_names)

    # 2. create initialization arguments for each environment
    env_fn_kwargs = []
    for i in range(num_envs):

        _cfg = deepcopy(cfg)  # make sure to clone
        _cfg.seed = _cfg.seed + i  # iterator and sampler depends on this

        # print(">>>", i)
        # print(_cfg.pretty_text)
        # print(len(img_name_splits[i]), len(img_names))

        # FIXME: maybe change how the devices are allocated
        # if there are multiple devices (cuda), it would be
        # nice to somehow split the devices evenly

        kwargs = dict(
            cfg=_cfg,
            filter_fn=partial(filter_by_name, names=img_name_splits[i]),
            num_episodes_per_img=-1,
            split=split,
            rank=i,
            dtype=dtype,
            device=device,
        )
        env_fn_kwargs.append(kwargs)

    # 3. initialize the vectorized environment
    if vec_type == "mp":
        envs = CorruptedMPVecEnv(
            make_env_fn=make_corrupted_rl_env_fn if is_rlenv else make_corrupted_env_fn,
            env_fn_kwargs=env_fn_kwargs,
            auto_reset_done=auto_reset_done,
        )
    elif vec_type == "equilib":
        # NOTE: faster than multiprocessing
        envs = CorruptedEquilibVecEnv(
            make_env_fn=make_corrupted_rl_env_fn if is_rlenv else make_corrupted_env_fn,
            env_fn_kwargs=env_fn_kwargs,
            auto_reset_done=auto_reset_done,
        )
    elif vec_type == "threaded":
        # NOTE: fastest by far
        envs = CorruptedThreadedVecEnv(
            make_env_fn=make_corrupted_rl_env_fn if is_rlenv else make_corrupted_env_fn,
            env_fn_kwargs=env_fn_kwargs,
            auto_reset_done=auto_reset_done,
        )
    else:
        raise ValueError(f"ERR: {vec_type} not supported")

    return envs
