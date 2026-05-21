import numpy as np
from typing import Any, Optional, Dict

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium import core, spaces

from dm_control import suite
from dm_env import specs

from envs.adapters.gymnasium import Gymnasium


class DMControl(Gymnasium):
    def __init__(
        self,
        env: gym.Env,
        env_params: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(env, env_params)

    @classmethod
    def create(cls, env_name: str, num_envs: int, **kwargs) -> "Gymnasium":
        domain_name, task_name = env_name.split("_")
        env = make(domain_name, task_name, num_envs=num_envs, **kwargs)
        return cls(env, env_params={"capacity": num_envs, **kwargs})


def _spec_to_box(spec, dtype):
    def extract_min_max(s):
        assert s.dtype == np.float64 or s.dtype == np.float32
        dim = int(np.prod(s.shape))
        if type(s) is specs.Array:
            bound = np.inf * np.ones(dim, dtype=np.float32)
            return -bound, bound
        elif type(s) is specs.BoundedArray:
            zeros = np.zeros(dim, dtype=np.float32)
            return s.minimum + zeros, s.maximum + zeros

    mins, maxs = [], []
    for s in spec:
        mn, mx = extract_min_max(s)
        mins.append(mn)
        maxs.append(mx)
    low = np.concatenate(mins, axis=0).astype(dtype)
    high = np.concatenate(maxs, axis=0).astype(dtype)
    assert low.shape == high.shape
    return spaces.Box(low, high, dtype=dtype)


def _flatten_obs(obs):
    obs_pieces = []
    for v in obs.values():
        flat = np.array([v]) if np.isscalar(v) else v.ravel()
        obs_pieces.append(flat)
    return np.concatenate(obs_pieces, axis=0)


class DMCtoGym(core.Env):
    def __init__(
        self,
        domain_name,
        task_name,
        task_kwargs=None,
        visualize_reward={},
        from_pixels=False,
        _render_mode="rgb_array",
        render_height=64,
        render_width=64,
        camera_id=0,
        frame_skip=1,
        environment_kwargs=None,
        channels_first=True,
    ):
        self.domain_name = domain_name
        self.task_name = task_name
        self.task_kwargs = task_kwargs or {}
        self.visualize_reward = visualize_reward
        self.environment_kwargs = environment_kwargs

        if "random" not in self.task_kwargs:
            self.task_kwargs["random"] = np.random.randint(0, 2**16)

        self.from_pixels = from_pixels
        self._render_mode = _render_mode
        self.render_height = render_height
        self.render_width = render_width
        self.camera_id = camera_id
        self.frame_skip = frame_skip
        self.channels_first = channels_first

        # create task
        self._env = suite.load(
            domain_name=domain_name,
            task_name=task_name,
            task_kwargs=task_kwargs,
            visualize_reward=visualize_reward,
            environment_kwargs=environment_kwargs,
        )

        # true and normalized action spaces
        self._true_action_space = _spec_to_box([self._env.action_spec()], np.float32)
        self._norm_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=self._true_action_space.shape, dtype=np.float32
        )

        # create observation space
        if from_pixels:
            shape = (
                [3, render_height, render_width]
                if channels_first
                else [render_height, render_width, 3]
            )
            self._observation_space = spaces.Box(
                low=0, high=255, shape=shape, dtype=np.uint8
            )
        else:
            self._observation_space = _spec_to_box(
                self._env.observation_spec().values(), np.float64
            )

        self._state_space = _spec_to_box(
            self._env.observation_spec().values(), np.float64
        )
        self.current_state = None

    def __getattr__(self, name):
        return getattr(self._env, name)

    def _get_obs(self, time_step):
        if self.from_pixels:
            obs = self.render()
            if self.channels_first:
                obs = obs.transpose(2, 0, 1).copy()
        else:
            obs = _flatten_obs(time_step.observation)
        return obs

    def _convert_action(self, action):
        action = action.astype(np.float64)
        true_delta = self._true_action_space.high - self._true_action_space.low
        norm_delta = self._norm_action_space.high - self._norm_action_space.low
        action = (action - self._norm_action_space.low) / norm_delta
        action = action * true_delta + self._true_action_space.low
        action = action.astype(np.float32)
        return action

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_space(self):
        return self._state_space

    @property
    def action_space(self):
        return self._norm_action_space

    @property
    def reward_range(self):
        return 0, self.frame_skip

    def step(self, action):
        assert self._norm_action_space.contains(action)
        action = self._convert_action(action)
        assert self._true_action_space.contains(action)
        reward = 0
        extra = {"internal_state": self._env.physics.get_state().copy()}

        for _ in range(self.frame_skip):
            time_step = self._env.step(action)
            reward += time_step.reward or 0
            done = time_step.last()
            termination = done and time_step.discount == 0.0
            truncation = done and time_step.discount != 0.0
            if done:
                break
        obs = self._get_obs(time_step)
        self.current_state = _flatten_obs(time_step.observation)
        extra["discount"] = time_step.discount
        return obs, reward, termination, truncation, extra

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.task_kwargs["random"] = seed
            self._env = suite.load(
                domain_name=self.domain_name,
                task_name=self.task_name,
                task_kwargs=self.task_kwargs,
                visualize_reward=self.visualize_reward,
                environment_kwargs=self.environment_kwargs,
            )
        time_step = self._env.reset()
        self.current_state = _flatten_obs(time_step.observation)
        obs = self._get_obs(time_step)
        info = {}
        return obs, info

    def render(self):
        assert self._render_mode == "rgb_array", (
            f"Only 'rgb_array' mode is supported; got {self._render_mode!r}"
        )
        return self._env.physics.render(
            height=self.render_height, width=self.render_width, camera_id=self.camera_id
        )


def make(
    domain_name,
    task_name,
    num_envs,
    vectorization_mode="sync",
    visualize_reward=False,
    from_pixels=False,
    render_mode="rgb_array",
    render_height=64,
    render_width=64,
    camera_id=0,
    frame_skip=1,
    episode_length=1000,
    environment_kwargs=None,
    time_limit=None,
    channels_first=True,
):
    env_id = "dmc_%s_%s-v1" % (domain_name, task_name)

    if from_pixels:
        assert not visualize_reward, (
            "cannot use visualize reward when learning from pixels"
        )

    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    if env_id not in gym.registry:
        task_kwargs = {}
        if time_limit is not None:
            task_kwargs["time_limit"] = time_limit

        register(
            id=env_id,
            entry_point=DMCtoGym,
            kwargs=dict(
                domain_name=domain_name,
                task_name=task_name,
                task_kwargs=task_kwargs,
                environment_kwargs=environment_kwargs,
                visualize_reward=visualize_reward,
                from_pixels=from_pixels,
                _render_mode=render_mode,
                render_height=render_height,
                render_width=render_width,
                camera_id=camera_id,
                frame_skip=frame_skip,
                channels_first=channels_first,
            ),
            max_episode_steps=max_episode_steps,
        )
    return gym.make_vec(
        env_id, num_envs=num_envs, vectorization_mode=vectorization_mode
    )
