from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any
from config import Env, Config, Wrapper


@dataclass
class EnvSelector:
    """The entry for selecting the env specific config, which requires knowing the env_name in the first place."""
    env: Env = field(default_factory=Env)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


FAMILY_OVERRIDES = {
    "gymnax": {},
    "mjx": {},
    "brax": {},
    "navix": {},
    "craftax": {},
    "envpool": {},
}


ENV_OVERRIDES = {
    # Examples
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
                "perception": {
                    "type": "linear" # state-based
                }
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
                "perception": {
                    "type": "cnn"    # pixel-based
                }
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

    env_family = env_id.split("/")[0] if "/" in env_id else None

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
    prefill: Wrapper = field(default_factory=Wrapper)

    def __call__(self) -> dict[str, Any]:
        return {k: v() for k, v in vars(self).items() if is_dataclass(v)}


def post_process(env_id: str, config: Config) -> Config:
    env_family = env_id.split("/")[0] if "/" in env_id else None
    if env_family == "envpool":
        if config.env.separated: # Multiple envs for different purposes
            separated_creations = {
                "train": {"vmap_multiplier": config.num_seeds, **config.env.creation},
                "eval": {"vmap_multiplier": config.num_seeds * config.exploration.num_eval_episodes, **config.env.creation},
                "prefill": {"vmap_multiplier": config.num_seeds * config.exploration.num_prefill_episodes, **config.env.creation},
            }
            separated_wrappers = Separated(
                train=replace(config.env.wrapper, num_envs=config.env.wrapper.num_envs),
                eval=replace(config.env.wrapper, num_envs=1),
                prefill=replace(config.env.wrapper, num_envs=config.env.wrapper.num_envs),
            )
            config.env.creation = separated_creations
            config.env.wrapper = separated_wrappers
        else:
            vmap_multiplier = config.num_seeds * max(config.exploration.num_prefill_episodes, config.exploration.num_eval_episodes) # Get upper bound so it can fit all
            config.env.creation["vmap_multiplier"] = vmap_multiplier
    return config
