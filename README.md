# Jaxinn: A Modular, Vectorized, and Extensible JAX Reinforcement Learning Framework

## Code Structure

```
Jaxinn/
├─ agent/
│  ├─ memory/
│  │  ├─ __init__.py
│  │  ├─ buffer.py
│  │  └─ storage.py
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ actor.py
│  │  ├─ critic.py
│  │  ├─ utils.py
│  │  └─ world.py
│  ├─ __init__.py
│  ├─ core.py
│  └─ utils.py
├─ envs/
│  ├─ adapters/
│  │  ├─ __init__.py
│  │  ├─ brax.py
│  │  ├─ craftax.py
│  │  ├─ dm_control.py
│  │  ├─ envpool.py
│  │  ├─ gymnasium.py
│  │  ├─ gymnax.py
│  │  ├─ mujoco_playground.py
│  │  └─ navix.py
│  ├─ __init__.py
│  ├─ environment.py
│  ├─ factory.py
│  ├─ spaces.py
│  ├─ vmap.py
│  └─ wrapper.py
├─ config.py
├─ custom.py
├─ logger.py
└─ train.py
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
- [ ] Jumanji
- [ ] JaxARC
- [ ] JaxGCRL

### Non-JAX Environments
- [X] EnvPool
- [X] Gymnasium
- [X] DeepMind Control Suite
- [ ] ARC-AGI-3
- [ ] MineRL

## Examples

## Contributing

## License

## Citation
