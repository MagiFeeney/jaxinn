import warnings
from dataclasses import dataclass
from typing import Dict, Any
import re
import importlib

from .wrapper import (
    Batched,
    TimeLimit,
    AutoReset,
    ActionRepeat,
    ChannelFirst,
    UnsqueezeScalar,
    OneHotAction,
    NextStepAutoResetTerminalObs,
    ResizeImage,
    NormalizeObservation,
    NormalizeReward,
    Branched,
)
from .environment import Environment
from .spaces import Discrete


@dataclass(frozen=True)
class EnvSpec:
    module: str
    cls_name: str
    channel_first: bool = False
    native_autoreset: bool = False
    native_time_limit: bool = True
    next_step_autoreset: bool = False


_FACTORY_REGISTRY = {
    # JAX envs
    "gymnax":    EnvSpec(".adapters.gymnax", "Gymnax", native_autoreset=True),
    "mjx":       EnvSpec(".adapters.mujoco_playground", "Playground", native_time_limit=False),
    "brax":      EnvSpec(".adapters.brax", "Brax", native_time_limit=False),
    "navix":     EnvSpec(".adapters.navix", "Navix"),
    "craftax":   EnvSpec(".adapters.craftax", "Craftax"),
    "jaxarc":    EnvSpec(".adapters.jaxarc", "JaxARC"),

    # Non-JAX envs
    "gymnasium": EnvSpec(".adapters.gymnasium", "Gymnasium", native_autoreset=True, next_step_autoreset=True),
    "dmc":       EnvSpec(".adapters.dm_control", "DMControl", channel_first=True, native_autoreset=True, next_step_autoreset=True),
    "envpool":   EnvSpec(".adapters.envpool", "EnvPool", channel_first=True, native_autoreset=True, next_step_autoreset=True),
    "arc":       EnvSpec(".adapters.arc", "ARC", channel_first=True, native_autoreset=True, next_step_autoreset=True),
    "maniskill": EnvSpec(".adapters.maniskill", "ManiSkill", native_autoreset=True),
}


def make_env(
        env_id: str,
        separated: bool,
        prefill_mode: str,
        creation: Dict[str, Any],
        wrapper: Dict[str, Any],
) -> Environment:
    parts = re.split(r'[:/]', env_id, maxsplit=1)
    if len(parts) != 2:
        raise ValueError(f"Invalid env_id '{env_id}'. Must be 'source/env_name'.")
    source, env_name = parts[0].lower(), parts[1]

    if source not in _FACTORY_REGISTRY:
        raise ValueError(f"Unknown source: '{source}'. Valid: {list(_FACTORY_REGISTRY.keys())}")

    spec = _FACTORY_REGISTRY[source]

    try:
        module = importlib.import_module(spec.module, package=__package__)
        cls = getattr(module, spec.cls_name)
    except (ImportError, TypeError) as e:
        raise ImportError(f"Failed to import adapter for '{source}'. Do you have '{spec.module}' installed?") from e

    def create_single_env(creation, wrapper):
        env = cls.create(env_name, **creation)

        if (not spec.channel_first) and len(env.observation_space.shape) > 1:
            env = ChannelFirst(env)

        if not spec.native_time_limit:
            max_episode_length = wrapper.get("max_episode_length", None)

            if max_episode_length is None:
                max_episode_length = getattr(env, "episode_length", None)

            if max_episode_length is None:
                max_episode_length = 1000
                warnings.warn(
                    "No episode length specified for truncation. "
                    "`max_episode_length` was not found in the wrapper config "
                    "or the environment attributes. "
                    f"Falling back to default: {max_episode_length}.",
                    stacklevel=2,
                )
            env = TimeLimit(env, max_episode_length)

        action_repeat = wrapper.get("action_repeat", 1)
        if action_repeat > 1:
            env = ActionRepeat(env, action_repeat)

        if not spec.native_autoreset:
            env = AutoReset(env)

        if spec.next_step_autoreset:
            env = NextStepAutoResetTerminalObs(env)

        use_one_hot_action = wrapper.get("use_one_hot_action", False)
        if isinstance(env.action_space, Discrete) and use_one_hot_action:
            env = OneHotAction(env)

        if wrapper.get("target_shape") is not None:
            env = ResizeImage(env, wrapper["target_shape"])

        env = UnsqueezeScalar(env)

        env = Batched(env, num_envs=wrapper["num_envs"])

        if wrapper.get("normalize_obs", False):
            env = NormalizeObservation(env)

        if wrapper.get("normalize_reward", False):
            env = NormalizeReward(env)

        return env

    if separated and isinstance(creation, dict) and isinstance(wrapper, dict):
        env = {
            mode: create_single_env(creation[mode], wrapper[mode])
            for mode in creation.keys()
        }
        if prefill_mode == "serial": # Share the same env as train if the prefill mode is serial
            env["prefill"] = env["train"]
    else:
        env = create_single_env(creation, wrapper)

    env = Branched(env, separated)         # For unified interface
    return env
