"""Atari environment implementing the OpenEnv interface.

Wraps the Arcade Learning Environment (ALE) to serve Atari games
via the standard reset/step/state API.
"""

import random
from typing import Optional
from uuid import uuid4

import numpy as np
from ale_py import ALEInterface, roms

from openenv.core.env_server import Environment

from atari_env.models import AtariAction, AtariObservation, AtariState


class AtariEnvironment(Environment[AtariAction, AtariObservation, AtariState]):
    """Atari game environment using ALE."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self._ale: Optional[ALEInterface] = None
        self._game_name = "pong"
        self._obs_type = "rgb"
        self._full_action_space = False
        self._frameskip = 4
        self._repeat_action_probability = 0.0
        self._episode_id = ""
        self._step_count = 0
        self._done = False

    def _setup_ale(self, game_name: str, obs_type: str, full_action_space: bool):
        """Initialize or reconfigure ALE."""
        if self._ale is not None and self._game_name == game_name:
            return

        self._game_name = game_name
        self._obs_type = obs_type
        self._full_action_space = full_action_space

        ale = ALEInterface()
        ale.setInt("random_seed", random.randint(0, 2**31))
        ale.setFloat("repeat_action_probability", self._repeat_action_probability)
        ale.setInt("frame_skip", self._frameskip)

        rom_path = getattr(roms, game_name.capitalize(), None)
        if rom_path is None:
            # Try common name variations
            for attr in dir(roms):
                if attr.lower() == game_name.lower():
                    rom_path = getattr(roms, attr)
                    break
        if rom_path is None:
            raise ValueError(f"ROM not found for game: {game_name}")

        ale.loadROM(rom_path)
        self._ale = ale

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs,
    ) -> AtariObservation:
        game_name = kwargs.get("game_name", "pong")
        obs_type = kwargs.get("obs_type", "rgb")
        full_action_space = kwargs.get("full_action_space", False)

        self._setup_ale(game_name, obs_type, full_action_space)

        if seed is not None:
            self._ale.setInt("random_seed", seed)

        self._ale.reset_game()
        self._episode_id = episode_id or str(uuid4())
        self._step_count = 0
        self._done = False

        return self._make_observation(reward=0.0)

    def step(
        self,
        action: AtariAction,
        timeout_s: Optional[float] = None,
        **kwargs,
    ) -> AtariObservation:
        if self._done:
            return self._make_observation(reward=0.0)

        legal = self._get_legal_actions()
        action_id = action.action_id
        if action_id not in legal:
            action_id = legal[0] if legal else 0

        reward = self._ale.act(action_id)
        self._step_count += 1
        self._done = self._ale.game_over()

        return self._make_observation(reward=float(reward))

    @property
    def state(self) -> AtariState:
        return AtariState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            game_name=self._game_name,
            obs_type=self._obs_type,
            full_action_space=self._full_action_space,
            frameskip=self._frameskip,
            repeat_action_probability=self._repeat_action_probability,
        )

    def _get_legal_actions(self):
        if self._full_action_space:
            return list(self._ale.getLegalActionSet())
        return list(self._ale.getMinimalActionSet())

    def _get_screen(self):
        if self._obs_type == "ram":
            return self._ale.getRAM().tolist(), [128]
        elif self._obs_type == "grayscale":
            screen = self._ale.getScreenGrayscale()
            return screen.flatten().tolist(), list(screen.shape)
        else:
            screen = self._ale.getScreenRGB()
            return screen.flatten().tolist(), list(screen.shape)

    def _make_observation(self, reward: float) -> AtariObservation:
        screen, screen_shape = self._get_screen()
        return AtariObservation(
            screen=screen,
            screen_shape=screen_shape,
            legal_actions=self._get_legal_actions(),
            lives=self._ale.lives(),
            episode_frame_number=self._ale.getEpisodeFrameNumber(),
            frame_number=self._ale.getFrameNumber(),
            done=self._done,
            reward=reward,
        )
