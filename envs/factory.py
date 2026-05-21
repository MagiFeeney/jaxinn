from dataclasses import dataclass
from typing import Dict, Any
import re
import importlib

from .wrapper import Batched, AutoReset, ActionRepeat, ChannelFirst, UnsqueezeScalar, OneHotAction, ResizeImage, Branched
from .environment import Environment


@dataclass(frozen=True)
class EnvSpec:
    module: str
    cls_name: str
    channel_first: bool = False
    native_batched: bool = True
    native_autoreset: bool = False


_FACTORY_REGISTRY = {
    # JAX envs
    "gymnax":    EnvSpec(".adapters.gymnax", "Gymnax", native_autoreset=True),
    "mjx":       EnvSpec(".adapters.mujoco_playground", "Playground"),
    "brax":      EnvSpec(".adapters.brax", "Brax"),
    "navix":     EnvSpec(".adapters.navix", "Navix"),
    "craftax":   EnvSpec(".adapters.craftax", "Craftax"),
    "envpool":   EnvSpec(".adapters.envpool", "EnvPool", channel_first=True, native_batched=True, native_autoreset=True),

    # Non-JAX envs
    "gymnasium": EnvSpec(".adapters.gymnasium", "Gymnasium", native_autoreset=True),
    "dmc":       EnvSpec(".adapters.dm_control", "DMControl", channel_first=True, native_autoreset=True)
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

        if wrapper.get("action_repeat"):
            env = ActionRepeat(env, wrapper["action_repeat"])

        if not spec.native_autoreset:
            env = AutoReset(env)

        if env.is_action_space_discrete:
            env = OneHotAction(env)

        if wrapper.get("target_shape") is not None:
            env = ResizeImage(env, wrapper["target_shape"])

        env = UnsqueezeScalar(env)

        if spec.native_batched:
            env = Batched(env, num_envs=wrapper["num_envs"])
        return env

    if separated:
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
