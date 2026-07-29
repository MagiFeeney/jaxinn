import numpy as np
from typing import Any

import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register
from arcengine import GameState
import arc_agi
from arc_agi.rendering import frame_to_rgb_array

from .gymnasium import Gymnasium


class ARC(Gymnasium):
    def __init__(
        self,
        env: gym.Env,
        env_params: dict[str, Any] | None = None,
    ):
        super().__init__(env, env_params)

    @classmethod
    def create(cls, env_name: str, num_envs: int, **kwargs) -> "ARC":
        env = make_env(env_name, num_envs=num_envs, **kwargs)
        return cls(env, env_params={"capacity": num_envs, **kwargs})


def _make_action_space(arc_action_space):
    return spaces.Dict({
        "action_id": spaces.Discrete(len(arc_action_space)),
        "x": spaces.Discrete(64),
        "y": spaces.Discrete(64)
    })


def _make_observation_space(arc_observation_space, render_mode=None):
    if render_mode == "rgb_array":
        return spaces.Box(low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
    return spaces.Box(low=0, high=255, shape=(1, 64, 64), dtype=np.uint8)


class ARCGymnasium(gym.Env):
    def __init__(self, game_id="ls20", _render_mode=None):
        super().__init__()
        self.game_id = game_id
        self._render_mode = _render_mode
        self.current_level = 1

        self.arc = arc_agi.Arcade()
        self.env = self.arc.make(self.game_id)

        self._arc_actions = self.env.action_space
        self._action_space = _make_action_space(self.env.action_space)
        self._observation_space = _make_observation_space(self.env.observation_space, _render_mode)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_level = 1

        arc_obs = self.env.reset()

        obs = self._get_obs(arc_obs.frame)
        info = {}
        return obs, info

    def step(self, action):
        arc_action = self._arc_actions[action["action_id"]]
        action_data = {}

        if arc_action.is_complex():
            action_data = {
                "x": action["x"],
                "y": action["y"]
            }

        arc_obs = self.env.step(arc_action, data=action_data)

        terminated = False
        reward = -0.01

        if arc_obs:
            if hasattr(arc_obs, 'levels_completed') and arc_obs.levels_completed > self.current_level:
                reward += 1.0 * self.current_level
                self.current_level = arc_obs.levels_completed

            if arc_obs.state == GameState.WIN:
                terminated = True
                reward += 10.0 * self.current_level
            elif arc_obs.state == GameState.GAME_OVER:
                terminated = True
                reward = -1.0

        obs = self._get_obs(arc_obs.frame)

        info = {}
        if terminated:
            scorecard = self.arc.get_scorecard()
            if scorecard:
                info["final_rhae_score"] = scorecard.score

        return obs, reward, terminated, False, info

    def _get_obs(self, frame: list[np.ndarray]):
        active_frame = frame[-1]
        if self._render_mode == "rgb_array":
            rgb_obs = frame_to_rgb_array(None, active_frame, scale=1).astype(self.observation_space.dtype)
            return rgb_obs.swapaxes(0, -1)
        else:
            obs = np.asarray(active_frame, dtype=self.observation_space.dtype)
            if obs.ndim == 2:
                obs = np.expand_dims(obs, axis=0)
            return obs

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space


def make_env(
        game_id,
        num_envs,
        vectorization_mode="sync",
        render_mode="rgb_array",
        max_episode_length=100,
):
    env_id = f"arc3_{game_id}-v1"

    if env_id not in gym.registry:
        register(
            id=env_id,
            entry_point=ARCGymnasium,
            kwargs=dict(
                game_id=game_id,
                _render_mode=render_mode,
            ),
            max_episode_steps=max_episode_length,
        )
    return gym.make_vec(
        env_id, num_envs=num_envs, vectorization_mode=vectorization_mode
    )
