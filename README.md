<div align="center">
<img src="https://github.com/MagiFeeney/jaxinn/blob/main/images/jaxinn-white.png" width="500"  alt="logo"></img>
</div>


# Jaxinn: A Modular, Vectorized, and Extensible JAX Reinforcement Learning Framework

Jaxinn is a high-performance, pure JAX Reinforcement Learning (RL) framework with a clean, intuitive, object-oriented API. It empowers researchers to train massively parallel agents in seconds on GPUs, unlocking significantly higher throughput for both model-based and model-free RL. Jaxinn provides a unified interface for a diverse suite of environments—all of which are fully JIT-compatible, even if they aren't natively written in JAX. With its modular architecture, you can easily build custom agents by plugging together existing components, allowing you to focus on your algorithm rather than rewriting training boilerplate.

## Why Jaxinn?
<div align="center">
<img src="https://github.com/MagiFeeney/jaxinn/blob/main/images/jaxinn-overview.png" width="800"  alt="overview"></img>
</div>

- 🛡️ Strongly-typed Configuration: A modular, object-oriented config system that routes components directly at the command line.
- ⚡ Vectorize Everywhere: Parallelize environments, episodes, and agents effortlessly by adjusting just a few simple parameters.
- 🔄 Unified Interface: A single, universal trainer loop runs all agents seamlessly within a self-sufficient environment ecosystem.
- 🌍 Diverse Environments: JAX-native? Good to go. `jit` or `vmap` breaks? No problem.
- 📊 Built-in Logger: Out-of-the-box multi-seed experiment tracking with TensorBoard.
- 🧩 Highly Modular: Grab a loss, a set of models, and a rule to build a custom agent.

## Get Started
You can use Jaxinn by cloning the repository and use `uv` to manage dependency:

``` bash
git clone https://github.com/MagiFeeney/jaxinn.git
cd jaxinn
pip install uv
```

`uv` allows us to select the environment we want to use:

``` bash
uv pip install -e .[gymnax] # Install Gymnax

uv pip install -e .[envpool] # Install EnvPool

uv pip install -e .[all] # Install all environments
```

## Algorithms
### Model-based RL
- [X] Dreamer
- [X] DreamerV2
- [ ] TD-MPC

### Model-based RL
- [X] PPO
- [X] SAC
- [ ] TD3

## Adapters
### JAX Native Environments
- [X] Brax
- [X] Craftax
- [X] Gymnax
- [X] MuJoCo Playground
- [X] Navix
- [X] JaxARC
- [ ] Jumanji
- [ ] JaxGCRL

### Non-JAX Environments
- [X] EnvPool
- [X] Gymnasium
- [X] DeepMind Control Suite
- [X] ARC-AGI-3
- [ ] MineRL

## Examples
- Create single agent and train it:
``` python
from jaxinn.trainer import Trainer, resolve_agent_config
from jaxinn.agent import Agent
from jaxinn.configs import Config

key = jax.random.PRNGKey(42)
key_agent, key_train = jax.random.split(key, 2)

config = Config()
trainer = Trainer.create(config)

agent_config = resolve_agent_config(config, trainer.env)
agent = Agent.create(agent_config, key=key_agent, memory_id=memory_id)
final_agent, (metrics, evaluation) = trainer(agent, key_train)
```

- Create multiple agents and train them simultaneously:
``` python
import equinox as eqx

@eqx.filter_jit
@eqx.filter_vmap
def make_train(agent, key):
    return trainer(agent, key)

def make_agent(key, memory_id):
    return Agent.create(agent_config, key=key, memory_id=memory_id)

agents = jax.vmap(make_agent)(keys_agent, memory_id)
final_agent, (metrics, evaluation) = make_train(agents, keys_train)
```

- Create an env and use it separately:
``` python
import jax
import jax.numpy as jnp
import equinox as eqx

from jaxinn.envs.adapters.gymnax import Gymnax
from jaxinn.envs.wrapper import UnsqueezeScalar, Batched


@eqx.filter_vmap
def train_one_episode(key, local_num_envs=None):
    local_num_envs = num_envs if local_num_envs is None else local_num_envs
    key_reset, key_scan = jax.random.split(key, 2)
    transition, env_info, env_state = env.reset(key_reset, local_num_envs)

    def make_episode_fn(carry, _):
        key, transition, env_state = carry
        key, key_step, key_action = jax.random.split(key, 3)
        action = jax.random.randint(key_action, (local_num_envs,), 0, env.action_space.n)
        transition, env_info, next_env_state = env.step(key_step, env_state, action)
        return (key, transition, next_env_state), (transition, env_state)

    _, (transitions, env_states) = jax.lax.scan(
        make_episode_fn,
        (key_scan, transition, env_state),
        None,
        num_steps_per_episode
    )
    return transitions, env_states


env = Gymnax.create(env_name, **creation)
env = UnsqueezeScalar(env)
env = Batched(env, num_envs=wrapper["num_envs"])
normalize_obs = wrapper.get("normalize_obs", False)
normalize_reward = wrapper.get("normalize_reward", False)
if normalize_obs or normalize_reward:
    env = Phase(env)
if normalize_obs:
    env = NormalizeObservation(env)
if normalize_reward:
    env = NormalizeReward(env)


key = jax.random.PRNGKey(42)
keys = jax.random.split(key, num_episodes)

print(" Running multiple episodes ... ")
transitions, env_states = train_one_episode(keys)
```

- Create a custom agent:
``` python
from typing import Any, Tuple, Dict

import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray
import equinox as eqx

from jaxinn.agent.rules import Agent, Learner
from jaxinn.agent.losses import CustomLossMixIn
from jaxinn.agent.models import ActorCritic
from jaxinn.agent.memory import Uniform as UniformMemory

from jaxinn.configs.agent import CustomAgentConfig


class CustomAgent(CustomLossMixIn, Agent):
    # Bind your agent to the config so that we can directly route it from cli.
    config_cls: ClassVar[Type] = CustomAgentConfig

    actor_critic: Learner[ActorCritic]
    memory: UniformMemory

    # Define your update rule and all set!
    def learn(self, data: Any, key: PRNGKeyArray) -> Tuple["Agent", Dict[str, jax.Array]]:
        pass
```

## Code Structure
<details>
<summary>Show details</summary>

```
Jaxinn/
├─ jaxinn/
│  ├─ __init__.py
│  ├─ logger.py
│  ├─ main.py
│  ├─ structs.py
│  ├─ trainer.py
│  ├─ agent/
│  │  ├─ losses/
│  │  │  ├─ __init__.py
│  │  │  ├─ base.py
│  │  │  ├─ model_based.py
│  │  │  ├─ model_free.py
│  │  │  └─ utils.py
│  │  ├─ memory/
│  │  │  ├─ __init__.py
│  │  │  ├─ buffer.py
│  │  │  └─ storage.py
│  │  ├─ models/
│  │  │  ├─ perception/
│  │  │  │  ├─ __init__.py
│  │  │  │  ├─ action_encoder.py
│  │  │  │  ├─ base.py
│  │  │  │  ├─ cnn.py
│  │  │  │  └─ linear.py
│  │  │  ├─ world/
│  │  │  │  ├─ __init__.py
│  │  │  │  ├─ representation.py
│  │  │  │  ├─ reward.py
│  │  │  │  ├─ transition.py
│  │  │  │  └─ world.py
│  │  │  ├─ __init__.py
│  │  │  ├─ actor.py
│  │  │  ├─ critic.py
│  │  │  ├─ distributions.py
│  │  │  ├─ heads.py
│  │  │  └─ utils.py
│  │  ├─ rules/
│  │  │  ├─ model_based/
│  │  │  │  ├─ __init__.py
│  │  │  │  └─ dreamer.py
│  │  │  ├─ model_free/
│  │  │  │  ├─ __init__.py
│  │  │  │  ├─ ppo.py
│  │  │  │  └─ sac.py
│  │  │  ├─ __init__.py
│  │  │  ├─ base.py
│  │  │  ├─ learner.py
│  │  │  └─ utils.py
│  │  ├─ __init__.py
│  │  └─ registry.py
│  ├─ configs/
│  │  ├─ agent/
│  │  │  ├─ __init__.py
│  │  │  ├─ base.py
│  │  │  ├─ dreamer.py
│  │  │  ├─ memory.py
│  │  │  ├─ ppo.py
│  │  │  └─ sac.py
│  │  ├─ __init__.py
│  │  ├─ base.py
│  │  ├─ config.py
│  │  ├─ custom.py
│  │  ├─ env.py
│  │  ├─ head.py
│  │  └─ model.py
│  ├─ envs/
│  │  ├─ adapters/
│  │  │  ├─ __init__.py
│  │  │  ├─ arc.py
│  │  │  ├─ brax.py
│  │  │  ├─ craftax.py
│  │  │  ├─ dm_control.py
│  │  │  ├─ envpool.py
│  │  │  ├─ gymnasium.py
│  │  │  ├─ gymnax.py
│  │  │  ├─ jaxarc.py
│  │  │  ├─ maniskill.py
│  │  │  ├─ mujoco_playground.py
│  │  │  └─ navix.py
│  │  ├─ __init__.py
│  │  ├─ environment.py
│  │  ├─ factory.py
│  │  ├─ spaces.py
│  │  ├─ vmap.py
└─ └─ └─ wrapper.py
```
</details>

## Contributing
Issues and PRs are welcome, but please be specific, considerate, and communicate your problem or goal clearly.

## License
Jaxinn is licensed under the [GNU General Public License v3.0 (GPLv3)](LICENSE) for greater accessibility.

## Citation
Should you find this work useful for your research, please consider citing:
``` bibtex
@software{jaxinn2026,
  title={{Jaxinn}: A Modular, Vectorized, and Extensible JAX Reinforcement Learning Framework},
  author={Jianfei Ma and Wee Sun Lee},
  url={https://github.com/MagiFeeney/jaxinn}
  year={2026},
}
```
