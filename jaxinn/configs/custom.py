from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any, Optional

from .env import Wrapper
from .config import Config


FAMILY_OVERRIDES = {
    "gymnax": {},
    "mjx": {},
    "brax": {},
    "navix": {},
    "craftax": {},
    "envpool": {},
}


ENV_OVERRIDES = {
    # Examples for illustration
    "gymnax/CartPole-v1": {
        "exploration": {
            "episode_length": 100,
            "action_noise": 0.5
        },
        "agent": {
            "actor": {
                "optimizer": {"lr": 3e-4}
            },
            "world": {
                "optimizer": {"lr": 3e-4},
            }
        }
    },
    "gymnax/DeepSea-bsuite": {
        "exploration": {
            "episode_length": 100,
            "action_noise": 0.5
        },
        "agent": {
            "actor": {
                "optimizer": {"lr": 3e-4}
            },
            "world": {
                "optimizer": {"lr": 3e-4},
            }
        }
    },
}


def get_config(env_id: str, custom_updates: dict = None) -> Config:
    """
    Generates a complete Config object tailored for a specific environment,
    applying family-level overrides first, then env-specific overrides, lastly custom overrides if any.
    """
    config = Config()

    env_family = env_id.split("/", maxsplit=1)[0] if "/" in env_id else None

    if env_family and env_family in FAMILY_OVERRIDES:
        config.update(FAMILY_OVERRIDES[env_family])

    if env_id in ENV_OVERRIDES:
        config.update(ENV_OVERRIDES[env_id])

    if custom_updates:
        config.update(custom_updates)

    return config


@dataclass
class Separated:
    train: Wrapper = field(default_factory=Wrapper)
    eval: Wrapper = field(default_factory=Wrapper)
    prefill: Optional[Wrapper] = None

    def __call__(self) -> dict[str, Any]:
        return {k: v() for k, v in vars(self).items() if v is not None and is_dataclass(v)}


def get_separate_env_config(config):
    train_num_envs = config.num_seeds * 1 * config.env.wrapper.num_envs
    eval_num_envs = config.num_seeds * config.exploration.num_eval_episodes * 1
    separated_creations = {
        "train": {"num_envs": train_num_envs, **config.env.creation},
        "eval": {"num_envs": eval_num_envs, **config.env.creation},
    }
    separated_wrappers_kwargs = {
        "train": replace(config.env.wrapper, num_envs=config.env.wrapper.num_envs),
        "eval": replace(config.env.wrapper, num_envs=1),
    }
    if config.env.prefill_mode == "batched":
        prefill_num_envs = config.num_seeds * config.exploration.num_prefill_episodes * config.env.wrapper.num_envs
        separated_creations["prefill"] = {"num_envs": prefill_num_envs, **config.env.creation}
        separated_wrappers_kwargs["prefill"] = replace(config.env.wrapper, num_envs=config.env.wrapper.num_envs)
    return separated_creations, Separated(**separated_wrappers_kwargs)


def post_process(env_id: str, config: Config) -> Config:
    config.exploration.action_repeat = config.env.wrapper.action_repeat
    config.exploration.prefill_mode = config.env.prefill_mode
    if config.env.prefill_mode == "external":
        raise NotImplementedError(
            "Prefill with external dataset is not implemented."
        )
    env_family = env_id.split("/", maxsplit=1)[0] if "/" in env_id else None
    if env_family in ("envpool", "mjx", "gymnasium", "dmc"):
        if config.env.separated: # Multiple envs for different purposes
            separated_creations, separated_wrappers = get_separate_env_config(config)
            config.env.creation = separated_creations
            config.env.wrapper = separated_wrappers
        else:
            num_envs = config.num_seeds * max(config.exploration.num_prefill_episodes, config.exploration.num_eval_episodes) * config.env.wrapper.num_envs # Get upper bound so it can fit all
            config.env.creation["num_envs"] = num_envs
    return config
