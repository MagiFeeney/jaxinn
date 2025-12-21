# Jaxinn

## Structure

```
Jaxinn/
├─ agent/
│  ├─ models/
│  │  ├─ actor.py
│  │  ├─ critic.py
│  │  └─ world.py
│  ├─ core.py
│  └─ memory.py
├─ envs/
│  └─ __init__.py
├─ README.md
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
