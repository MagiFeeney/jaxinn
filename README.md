<div align="center">
<img src="https://github.com/MagiFeeney/jaxinn/blob/main/images/jaxinn-white.png" width="500"  alt="logo"></img>
</div>


# Jaxinn: A Modular, Vectorized, and Extensible JAX Reinforcement Learning Framework

Jaxinn is a high-performance, pure JAX Reinforcement Learning (RL) framework with a clean, intuitive, object-oriented API. It empowers researchers to train massively parallel agents in seconds on GPUs, unlocking significantly higher throughput for both model-based and model-free RL. Jaxinn provides a unified interface for a diverse suite of environments—all of which are fully JIT-compatible, even if they aren't natively written in JAX. With its modular architecture, you can easily build custom agents by plugging together existing components, allowing you to focus on your algorithm rather than rewriting training boilerplate.

## Why Jaxinn?
<div align="center">
<img src="https://github.com/MagiFeeney/jaxinn/blob/main/images/jaxinn-overview.png" width="800"  alt="overview"></img>
</div>

- Point 1
- Point 2
- Point 3

## Get Started

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

## Contributing

## License

## Citation
