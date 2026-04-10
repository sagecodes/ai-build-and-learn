"""Pydantic models for the Atari OpenEnv environment."""

from typing import List, Literal, Optional

from openenv.core.env_server import Action, Observation, State
from pydantic import Field


class AtariAction(Action):
    """Action: select a joystick direction / button."""
    action_id: int = 0
    game_name: str = "pong"
    obs_type: Literal["rgb", "grayscale", "ram"] = "rgb"
    full_action_space: bool = False


class AtariObservation(Observation):
    """What the agent sees — flattened screen pixels + game info."""
    screen: List[int] = Field(default_factory=list)
    screen_shape: List[int] = Field(default_factory=list)
    legal_actions: List[int] = Field(default_factory=list)
    lives: int = 0
    episode_frame_number: int = 0
    frame_number: int = 0


class AtariState(State):
    """Metadata about the current Atari session."""
    game_name: str = "pong"
    obs_type: str = "rgb"
    full_action_space: bool = False
    mode: Optional[int] = None
    difficulty: Optional[int] = None
    repeat_action_probability: float = 0.0
    frameskip: int = 4
