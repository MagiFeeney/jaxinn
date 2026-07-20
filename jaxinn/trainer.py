from typing import Any, Dict, Generic, Literal, Optional, Tuple, TypeVar

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import PRNGKeyArray

from jaxinn.structs import Experience, LatentState
from jaxinn.agent import Agent
from jaxinn.envs import Environment, make_env
from jaxinn.envs.wrapper import NextStepAutoResetTerminalObs, NormalizeObservation
from jaxinn.envs.spaces import Space, ComplexSpace, Discrete, OneHotDiscrete
from jaxinn.configs import AgentConfig, Config
from jaxinn.logger import JaxLogger as Logger


EnvState = TypeVar("EnvState")


class InteractionState(eqx.Module, Generic[EnvState]):
    experience: Experience
    latent_state: LatentState
    env_state: EnvState


InteractionMode = Literal["train", "prefill", "eval"]


class Interactor:
    @staticmethod
    def resolve_mode(eval: bool, prefill: bool) -> InteractionMode:
        if eval:
            return "eval"
        if prefill:
            return "prefill"
        return "train"

    def resolve_num_envs(self, mode: InteractionMode, num_envs: int | None) -> int:
        if num_envs is None or self.env.separated:
            return self.env.get("num_envs", mode)
        return num_envs

    def resolve_episode_length(self, num_envs: int, mode: InteractionMode) -> int:
        if mode == "eval":
            return self.env.get("max_episode_length") // self.action_repeat + 1
        episode_length = self.episode_length // num_envs // self.action_repeat
        if mode == "prefill" and self.prefill_mode == "serial":
            episode_length *= self.num_prefill_episodes
        return episode_length

    def init_interaction_state(
            self,
            agent: Agent,
            key: PRNGKeyArray,
            eval: bool = False,
            prefill: bool = False,
            num_envs: int | None = None,
            last_env_state: EnvState | None = None,
    ) -> InteractionState:
        mode = self.resolve_mode(eval, prefill)
        key_reset, key_init = jax.random.split(key, 2)

        if num_envs is None or self.env.separated:
            init_transition, info, env_state = self.env.reset(key_reset, mode=mode)
            num_envs = self.env.get("num_envs", mode)
        else:
            init_transition, info, env_state = self.env.reset(key_reset, num_envs=num_envs, mode=mode)
        init_latent_state = agent.init_latent_state(key_init, batch_shape=(num_envs,), eval=eval)
        init_terminal_obs = info.terminal_observation

        if last_env_state is not None:
            if self.env.is_wrapped_by(NormalizeObservation, mode=mode):
                env_state = eqx.tree_at(lambda s: s.obs_rms, env_state, last_env_state.obs_rms)

        if hasattr(env_state, "is_training") and eval:
            env_state = eqx.tree_at(lambda s: s.is_training, env_state, False)

        interaction_state = InteractionState(
            experience = Experience(
                transition=init_transition,
                terminal_observation=init_terminal_obs
            ),
            latent_state=init_latent_state,
            env_state=env_state
        )
        return interaction_state

    def interact(
            self,
            agent: Agent,
            interaction_state: InteractionState,
            key: PRNGKeyArray,
            eval: bool = False,
            prefill: bool = False,
            num_envs: int | None = None
    ) -> Tuple[InteractionState, Experience]:
        mode = self.resolve_mode(eval, prefill)
        num_envs = self.resolve_num_envs(mode, num_envs)
        episode_length = self.resolve_episode_length(num_envs, mode)

        def random_act_branch(operand):
            last_latent_state, _, _, key = operand
            keys = jax.random.split(key, num_envs)
            action = jax.vmap(self.env.get("action_space", mode).sample)(keys)
            return last_latent_state, action # For consistency

        def agent_act_branch(operand):
            last_latent_state, last_action, obs, key = operand
            key_act, key_noise = jax.random.split(key, 2)
            latent_state, action = agent.act(last_latent_state, last_action, obs, key=key_act, eval=eval)

            def apply_noise(action, action_space, key):
                if isinstance(action_space, OneHotDiscrete):
                    key_idx, key_cond = jax.random.split(key, 2)
                    random_idx = jax.random.randint(key_idx, action.shape[:-1], 0, action_space.size)
                    expl_action = jax.nn.one_hot(random_idx, action_space.size)

                    should_explore = jax.random.uniform(key_cond, (*action.shape[:-1], 1)) < self.action_noise
                    action = jnp.where(should_explore, expl_action, action)
                elif isinstance(action_space, Discrete):
                    key_idx, key_cond = jax.random.split(key, 2)
                    expl_action = jax.random.randint(key_idx, action.shape, 0, action_space.size)
                    should_explore = jax.random.uniform(key_cond, action.shape) < self.action_noise
                    action = jnp.where(should_explore, expl_action, action)
                else:
                    noise = jax.random.normal(key, shape=action.shape) * self.action_noise
                    action = jnp.clip(action + noise, -1.0, 1.0)
                return action

            if not eval and self.action_noise > 0:
                action_leaves, treedef = jax.tree.flatten(action)
                space_leaves = jax.tree.leaves(
                    self.env.get("action_space"),
                    is_leaf=lambda x: isinstance(x, Space) and not isinstance(x, ComplexSpace)
                )
                assert len(action_leaves) == len(space_leaves)

                keys = jax.random.split(key_noise, len(action_leaves))
                noised = [apply_noise(a, s, k) for a, s, k in zip(action_leaves, space_leaves, keys)]
                action = jax.tree.unflatten(treedef, noised)
            return latent_state, action

        def interact_step_fn(carry, _):
            interaction_state, key = carry
            last_transition = interaction_state.experience.transition
            key, key_init, key_action, key_step = jax.random.split(key, 4)

            # Isolate the reset observation in current-step autoreset environments.
            done = last_transition.terminated | last_transition.truncated
            mask = 1 - done
            last_latent_state = jax.tree.map(
                lambda current, reset: jnp.where(mask, current, reset),
                interaction_state.latent_state,
                agent.init_latent_state(key_init, batch_shape=(num_envs,))
            )
            last_action = jax.tree.map(
                lambda x: x * mask.reshape(mask.shape[:x.ndim] + (1,) * (x.ndim - mask.ndim)),
                last_transition.action
            )
            obs = last_transition.next_obs

            operand = (last_latent_state, last_action, obs, key_action)
            if prefill:
                latent_state, action = random_act_branch(operand)
            else:
                latent_state, action = agent_act_branch(operand)

            env_state = interaction_state.env_state
            transition, info, next_env_state = self.env.step(key_step, env_state, action, mode=mode)

            # Decouple terminal and reset observation in next-step autoreset environments.
            last_done = getattr(env_state, "last_done", None)
            if last_done is not None:
                latent_state = jax.tree.map(
                    lambda reset, current: jnp.where(last_done, reset, current),
                    agent.init_latent_state(key_init, batch_shape=(num_envs,)),
                    latent_state
                )
            new_interaction_state = InteractionState(
                experience = Experience(
                    transition=transition,
                    terminal_observation=info.terminal_observation
                ),
                latent_state=latent_state,
                env_state=next_env_state
            )

            return (new_interaction_state, key), interaction_state.experience

        (interaction_state, _), experiences = jax.lax.scan(
            interact_step_fn,
            (interaction_state, key),
            None,
            episode_length
        )
        return interaction_state, experiences


class Trainer(Interactor, eqx.Module):
    env: Environment = eqx.field(static=True)
    logger: Logger = eqx.field(static=True)

    num_environment_steps: int = eqx.field(static=True)
    num_prefill_episodes: int = eqx.field(static=True)
    eval_interval: int = eqx.field(static=True)
    train_interval: int = eqx.field(static=True)
    train_iterations: int = eqx.field(static=True)
    pretrain_iterations: int = eqx.field(static=True)
    episode_length: int = eqx.field(static=True)
    num_eval_episodes: int = eqx.field(static=True)
    action_repeat: int = eqx.field(static=True)
    action_noise: float = eqx.field(static=True)
    prefill_mode: str = eqx.field(static=True)
    restart: bool = eqx.field(static=True)

    @classmethod
    def create(cls, config: Config):
        env = make_env(**config.env(), wrapper=config.env.wrapper())
        logger = Logger.create(**config.logger(), axis_name=config.axis_name) if config.logger.log_dir else None
        return cls(env=env, logger=logger, **config.exploration())

    def __call__(self, agent: Agent, key: PRNGKeyArray) -> Tuple[Agent, Tuple[Dict[str, Any], jax.Array]]:
        key_prefill, key_interleaved = jax.random.split(key, 2)

        # Prefill
        if self.num_prefill_episodes > 0:
            agent, interaction_state, prefill_metrics = self.prefill(agent, key_prefill) # TODO: handle external dataset

            if self.logger and prefill_metrics:
                self.logger.log_dict(
                    prefill_metrics,
                    step=self.num_prefill_episodes * self.episode_length
                )

        # Train and evaluate
        def interleaved_step_fn(carry, iteration): # Evaluation truck with unit being Training truck
            agent, interaction_state, key = carry
            key, key_train, key_eval = jax.random.split(key, 3)

            # Training
            (agent, interaction_state, _), train_metrics = jax.lax.scan(
                lambda carry, _: self.train(*carry),
                (agent, interaction_state, key_train),
                None,
                self.eval_interval // self.train_interval
            )

            start_step = iteration * self.eval_interval + self.num_prefill_episodes * self.episode_length
            if self.logger and train_metrics:
                self.logger.log_sequence(
                    train_metrics,
                    start_step=start_step,
                    interval=self.train_interval
                )

            # Evaluation
            keys_eval = jax.random.split(key_eval, self.num_eval_episodes)
            last_env_state = interaction_state.env_state # Useful for cross-env data sharing
            episodic_returns = jax.vmap(lambda k: self.evaluate(agent, k, last_env_state=last_env_state))(keys_eval) # Parallel evaluation
            evaluation = jnp.mean(episodic_returns)

            eval_metrics = {"eval/mean": evaluation}
            eval_step = self.eval_interval + start_step
            if self.logger and eval_metrics:
                self.logger.log_dict(
                    eval_metrics,
                    step=eval_step,
                )

            extra = {"eval/return": episodic_returns}
            print_metrics = train_metrics | eval_metrics | extra
            if self.logger and print_metrics:
                self.logger.print_summary(
                    print_metrics,
                    step=eval_step,
                    headline_params={"n": self.num_eval_episodes}
                )

            return (agent, interaction_state, key), (train_metrics, evaluation)

        if self.prefill_mode != "serial" or self.restart:
            key, key_init = jax.random.split(key, 2)
            interaction_state = self.init_interaction_state(agent, key_init)

        (final_agent, _, _), (metrics, evaluation) = jax.lax.scan(
            interleaved_step_fn,
            (agent, interaction_state, key_interleaved),
            jnp.arange(self.num_environment_steps // self.eval_interval),
        )

        return final_agent, (metrics, evaluation)

    def learn(self, agent: Agent, key: PRNGKeyArray, prefill: bool = False) -> Tuple[Agent, Optional[Dict[str, jax.Array]]]:

        make_batch = agent.make_batch_fn()

        def learn_step_fn(carry, _):
            agent, key = carry
            key, key_batch, key_learn = jax.random.split(key, 3)
            data = make_batch(key_batch)
            new_agent, metrics = agent.learn(data, key_learn)
            return (new_agent, key), metrics

        num_iterations = self.train_iterations if not prefill else self.pretrain_iterations

        if num_iterations > 0:
            (agent, _), metrics = jax.lax.scan(
                learn_step_fn,
                (agent, key),
                None,           # TODO: add pre-computing option
                num_iterations
            )
            avg_metrics = jax.tree.map(jnp.mean, metrics)
            return agent, avg_metrics
        return agent, {}

    def prefill(self, agent: Agent, key: PRNGKeyArray) -> Tuple[Agent, InteractionState, Dict[str, jax.Array]]:
        key_init, key_interact, key_learn = jax.random.split(key, 3)

        def prefill_fn(key):
            key, key_init, key_interact = jax.random.split(key, 3)
            interaction_state = self.init_interaction_state(agent, key_init, prefill=True)
            interaction_state, experiences = self.interact(agent, interaction_state, key_interact, prefill=True)
            return interaction_state, experiences

        if self.prefill_mode == "batched":
            keys = jax.random.split(key_interact, self.num_prefill_episodes)
            interaction_state, experiences = jax.vmap(prefill_fn)(keys)
            source = 2
        elif self.prefill_mode == "serial":
            interaction_state, experiences = prefill_fn(key_interact)
            source = 1
        else:
            raise ValueError(f"Unknown prefill_mode: {self.prefill_mode}. Expected 'batched' or 'serial'.")
        agent = agent.add_experience(experiences, source=source)

        agent, metrics = self.learn(agent, key_learn, prefill=True)
        return agent, interaction_state, metrics

    def train(
            self,
            agent: Agent,
            interaction_state: InteractionState,
            key: PRNGKeyArray,
    ) -> Tuple[Tuple[Agent, InteractionState, PRNGKeyArray], Dict[str, jax.Array]]:
        key, key_interact, key_learn = jax.random.split(key, 3)
        keys_interact = jax.random.split(key_interact, self.train_interval // self.episode_length)
        interaction_state, experiences = jax.lax.scan(lambda s, k: self.interact(agent, s, k), interaction_state, keys_interact)

        # Store them
        agent = agent.add_experience(experiences, source=2)

        # Update
        agent, metrics = self.learn(agent, key_learn)
        return (agent, interaction_state, key), metrics

    def evaluate(self, agent: Agent, key: PRNGKeyArray, num_envs: int = 1, last_env_state: EnvState | None = None) -> jax.Array:
        key_init, key_interact = jax.random.split(key, 2)
        interaction_state = self.init_interaction_state(agent, key_init, eval=True, num_envs=num_envs, last_env_state=last_env_state)
        _, experiences = self.interact(agent, interaction_state, key_interact, eval=True, num_envs=num_envs)
        done = experiences.transition.terminated | experiences.transition.truncated
        masks = 1 - jnp.maximum.accumulate(done, axis=0)
        shifted_masks = jnp.concatenate([jnp.ones_like(masks[0:1]), masks[:-1]])
        cumulative_rewards = jnp.sum(experiences.transition.reward * shifted_masks) # Return up to the first termination inclusively
        return cumulative_rewards

    def close(self):
        if hasattr(self.logger, "close"):
            self.logger.close()
        if hasattr(self.env, "close"):
            self.env.close()

    def __enter__(self) -> "Trainer":
        return self

    def __exit__(self, *_) -> None:
        self.close()


def resolve_agent_config(config: Config, env: Environment) -> AgentConfig:
    env_wrapper_config = config.env.wrapper.train() if config.env.separated else config.env.wrapper()
    next_step_autoreset = env.is_wrapped_by(NextStepAutoResetTerminalObs)
    ctx = {       # TODO: perhaps pass the whole exploration for future compatibility
        "observation_space":        env.get("observation_space"),
        "action_space":             env.get("action_space"),
        "num_environment_steps":    config.exploration.num_environment_steps,
        "episode_length":           config.exploration.episode_length,
        "train_iterations":         config.exploration.train_iterations,
        "num_seeds":                config.num_seeds,
        "next_step_autoreset":      next_step_autoreset,
        **env_wrapper_config
    }
    return config.agent.resolve(ctx)
