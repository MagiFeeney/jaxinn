# Jaxinn

## Structure

```
Jaxinn/
├─ agent/
│  ├─ models/
│  │  ├─ actor.py
│  │  ├─ critic.py
│  │  └─ RSSM.py
│  ├─ core.py
│  └─ memory.py
└─ envs/
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
