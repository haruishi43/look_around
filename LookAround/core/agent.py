#!/usr/bin/env python3

from typing import Any, Dict, Union


class Agent:
    name: str

    def reset(self) -> None:
        """Called before starting a new episode in environment."""
        raise NotImplementedError

    def act(
        self,
        observations,
    ) -> Union[int, str, Dict[str, Any]]:
        """Called to produce an action to perform in an environment."""
        raise NotImplementedError
