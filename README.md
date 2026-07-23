<div align="center">
<img src="https://raw.githubusercontent.com/MagiFeeney/jaxinn/blob/main/images/jaxinn-black.png" alt="logo"></img>
</div>


# Jaxinn: A Modular, Vectorized, and Extensible JAX Reinforcement Learning Framework

## Code Structure

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

## Why Jaxinn?

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

## Contributing

## License

## Citation
