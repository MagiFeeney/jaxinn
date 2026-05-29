import os
from typing import Optional

import tyro
from rich import print as rprint
from rich.panel import Panel
import equinox as eqx
import jax
import jax.numpy as jnp

from trainer import Trainer, resolve_agent_config
from agent import Agent
from custom import EnvSelector, get_config, post_process
from config import Config


def setup_context(vectorization_mode: Optional[str]):
    if vectorization_mode == "async":
        try:
            import multiprocessing as mp
            if mp.get_start_method(allow_none=True) != 'spawn':
                mp.set_start_method('spawn')
        except RuntimeError:
            pass
    else:
        print("Skipping 'multiprocessing' setup: Environment is JAX-native or in 'sync' mode.")


def main(config):
    # Distribute RNG keys
    key = jax.random.PRNGKey(config.seed)
    num_devices = jax.device_count()
    if config.num_seeds % num_devices != 0:
        closest_lower = (config.num_seeds // num_devices) * num_devices
        closest_higher = closest_lower + num_devices
        raise ValueError(
            f"Mismatch: config.num_seeds ({config.num_seeds}) is not divisible by "
            f"num_devices ({num_devices}). \n"
            f"Please set --num_seeds to {closest_lower} or {closest_higher}."
        )
    num_seeds_per_device = config.num_seeds // num_devices
    keys = jax.random.split(key, config.num_seeds * 2)
    keys_agent, keys_train = keys.reshape(2, num_devices, num_seeds_per_device, -1)
    memory_ids = jnp.arange(num_devices * num_seeds_per_device).reshape(num_devices, num_seeds_per_device) # For anchoring cpu memory if enabled

    # Initialize trainer with environment
    trainer = Trainer.create(config)

    # Resolve agent config with environment-specific information
    agent_config = resolve_agent_config(config, trainer.env)

    # Spawn parallel agents
    def make_agent(key, memory_id):
        return Agent(agent_config, key=key, memory_id=memory_id)

    agents = jax.vmap(jax.vmap(make_agent))(keys_agent, memory_ids)

    # Load agent from a checkpoint
    if bool(config.load_model_path):
        agents = eqx.tree_deserialise_leaves(config.load_model_path, agents)

    # Ready to train
    @eqx.filter_pmap(
        axis_name=config.axis_name,
        donate="all"
    )                              # shard across devices, donate buffer for memory efficiency
    @eqx.filter_vmap               # vectorise within each device
    def make_train(agent, key):
        return trainer(agent, key)

    final_agent, (metrics, evaluation) = make_train(agents, keys_train)
    final_eval_return = evaluation.reshape(config.num_seeds, -1)[:, -1]

    # Get statistics of the final eval return
    mean = final_eval_return.mean().item()
    std = final_eval_return.std().item()
    raw_str = jnp.array_str(final_eval_return, precision=4, suppress_small=True)

    # Training summary
    summary_message = (
        f"[bold green]🚀 Training Completed Successfully![/bold green]\n\n"
        f"[cyan]Agents/Seeds:[/cyan]   [bold white]{config.num_seeds}[/bold white]\n"
        f"[cyan]Achieved Return:[/cyan] [bold yellow]{mean:.4f} ± {std:.4f}[/bold yellow]  "
        f"[dim white](Raw: {raw_str})[/dim white]"
    )
    rprint(
        Panel(
            summary_message,
            title="[bold magenta]Run Summary",
            border_style="green",
            expand=False
        )
    )

    # Save the final agent
    if bool(config.save_model_path):
        save_dir = os.path.dirname(config.save_model_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        eqx.tree_serialise_leaves(config.save_model_path, final_agent)

    # Close the trainer
    trainer.close()


if __name__ == "__main__":
    # Grab the env id
    env_selector, _ = tyro.cli(
        EnvSelector,
        return_unknown_args=True
    )
    env_id = env_selector.env_id

    # Final CLI Pass
    config = tyro.cli(
        Config,
        default=get_config(env_id)
    )

    # Setup multiprocessing context if the env is cpu-based
    setup_context(config.env.creation.get("vectorization_mode"))

    # Post processing
    config = post_process(env_id, config)

    # Run
    main(config)
