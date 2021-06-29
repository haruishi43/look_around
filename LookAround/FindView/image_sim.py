#!/usr/bin/env python3

from typing import (
    Dict,
    List,
    Any,
    Optional,
    Union,
)

import numpy as np


class Loader:
    """Manages Panorama Images
    """
    def __init__(
        self,
        config,
    ) -> None:
        """
        """
        self.config = config

        if self.config.VIDEO_FORMAT == 'JPG':
            from pano_env.core.loader.frame_loader import FrameLoader as Loader
        elif self.config.VIDEO_FORMAT == 'MP4':
            from pano_env.core.loader.video_loader import VideoLoader as Loader

        self.frame_count = 0
        self.loader = Loader(
            data_path = self.config.VIDEO,
            total_frames = self.config.TOTAL_FRAMES,
            nthreads = self.config.LOADER.NTHREADS
        )

    def set_position(self, position: int = 0):
        """
        :property position: frame position
        """
        self.loader.set_position(position)

    def step(self) -> np.ndarray:
        """Loads the next frame
        """
        frame = next(self.loader)
        self.frame_count += 1
        return frame

    def reset(self) -> np.ndarray:
        """Reset loader and counter and return frame 0
        """
        self.loader.reset()
        self.frame_count = 0
        frame = next(self.loader)
        return frame

    def did_reach_limit(self) -> int:
        """Checks if frame limit or end is reached
        """
        # FIXME: might be a mismatch between int and bool
        if self.frame_count >= self.config.FRAME_LIMIT:
            return 1
        elif self.loader.did_end():
            return 1
        else:
            return 0

    def get_time(self) -> float:
        """
        Return:
            time in float
        """
        return self.loader.get_time()

    def get_position(self) -> int:
        """
        Return:
            frame position in int
        """
        return self.loader.get_position()


class RotationState:
    """Tracks the agent's rotation
    """
    def __init__(self, config: Config) -> None:
        """
        :property config: Simulation Config file
        """
        self.config = config
        self.agent_cfg = self.config.AGENT

        # initialize rotation tracker:
        self.rot_tracker = RotationTracker(
            inc=self.agent_cfg.INCREMENTATION,
            pitch_thresh=self.agent_cfg.PITCH_THRESH,
        )

    def step(self, action: int) -> List[float]:
        """Return tracked radian angle that is updated by action
        """
        self.rot_tracker.update(action)
        rad = self.rot_tracker.rad
        return rad

    def reset(self) -> List[float]:
        """Reset tracker
        """
        self.rot_tracker.reset()
        return self.rot_tracker.rad

    def set_rotation(self, rad: List[float]):
        self.rot_tracker.rad = rad


class BaseSim:
    """Base simulation object
    """
    def __init__(self, config: Config):
        """
        :property config: Simulation Config file
        """
        self.config = config
        self.agent_cfg = config.AGENT

        self._pano2pers = Pano2Pers(self.config)
        self._pano_loader = PanoLoader(self.config)
        self._rot_tracker = RotationState(self.config)

        self.set_state()
        self._reload_after_setting_state()

        # TODO: better ways to manage observations?
        # use self._sensor = {}
        # Define SimSensor and create from Config
        # for each sensor, get_observations()?
        # slam won't work as sensor since it needs perspective image
        self.observations = {
            'rgb': None,
            'time': None,
            'frame_state': None,
            'slam_state': None,
            'keyframe_points': None,
        }

    def step(self, action: int) -> Dict[str, Any]:
        pano = self._pano_loader.step()
        rad = self._rot_tracker.step(action)
        pers = self._pano2pers.step(rad, pano)
        time = self.get_time()

        # update state
        self._state.position = self._pano_loader.get_position()
        self._state.rotation = rad

        # update observation
        self.observations['rgb'] = pers
        self.observations['time'] = time
        self.observations['frame_state'] = self._pano_loader.did_reach_limit()

        # slam
        if self.config.SLAM_ON:
            self._slam.process(pers, time)
            self.observations['slam_state'] = self._slam.get_slam_state()
            self.observations['keyframe_points'] = self._slam.get_keyframe_points()

        return self.observations

    def get_observation(self) -> Dict[str, Any]:
        return self.observations

    def get_state(self) -> AgentState:
        return self._state

    def get_time(self) -> float:
        return self._pano_loader.get_time()

    # ---------------------------------------------------------------------------
    # FIXME: Hacky...
    # ---------------------------------------------------------------------------
    def get_frame_state(self) -> bool:
        return bool(self._pano_loader.did_reach_limit())

    def get_slam_state(self):
        if self.config.SLAM_ON:
            return self._slam.get_slam_state()
        return None

    # ---------------------------------------------------------------------------
    # END: Hacky...
    # ---------------------------------------------------------------------------

    def set_state(self) -> None:
        self._state = AgentState(
            position = self.agent_cfg.START_POSITION,
            rotation = self.agent_cfg.START_ROTATION,
        )

    def _reload_after_setting_state(self):
        self._rot_tracker.set_rotation(self._state.rotation)
        self._pano_loader.set_position(self._state.position)

    def reset(self, config: Config) -> Dict[str, Any]:
        self.config = config
        self.agent_cfg = config.AGENT

        rot = self._rot_tracker.reset()

        self.set_state()
        self._reload_after_setting_state()

        pano = self._pano_loader.reset()
        pers = self._pano2pers.step(rot, pano)
        time = self.get_time()

        self.observations['rgb'] = pers
        self.observations['time'] = time
        self.observations['frame_state'] = self._pano_loader.did_reach_limit()

        return self.observations

    def seed(self, seed):
        pass

    def reconfigure(self, config: Config):
        self.config = config

        # i). need to change video
        self._pano_loader = PanoLoader(self.config)

        # ii). need to restart orbslam
        if self.config.SLAM_ON:
            self._slam.reset()

        # iii). set all observations to None
        self.observations = {
            'rgb': None,
            'time': None,
            'frame_state': None,
            'slam_state': None,
            'keypoints': None,
        }

    def close(self):
        del self._slam
        del self._pano2pers
        del self._pano_loader
        del self._rot_tracker
