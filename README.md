# Jaxinn

## Structure

```
Jaxinn/
├─ agent/
│  ├─ models/
│  │  ├─ __init__.py
│  │  ├─ actor.py
│  │  ├─ critic.py
│  │  ├─ utils.py
│  │  └─ world.py
│  ├─ __init__.py
│  ├─ core.py
│  └─ memory.py
├─ envs/
│  ├─ __init__.py
│  ├─ environment.py
│  └─ wrapper.py
├─ config.py
├─ README.md
├─ requirements.txt
└─ train.py
```

## Prototyping
- observe
  - interaction
  - inference
- imagine
  - rollout
    - vmap: batch operator
- update
  - model
    - jax parameter update
  - policy
    - jax parameter update
  - value
    - jax parameter update
  - return
    - jax scan
  - all in one
    - jax lax fori loop
- interaction
  - parallel envs
  - batched loop

## TODO
- [ ] Wrap the env to get the next real obs that is overwritten by the reset one.
- [ ] Handle the last time step being terminated, which is meaningless when sampled.
- [ ] Memory shape should follow the env shape.
  - [ ] Flatten according to env axis, so different trajectories are stacked in order.
