""" 
PPO-RE (Reward Exchange) on Cleanup.

Based on the IPPO CNN Cleanup implementation, with a reward-exchange
mechanism:

    U_i(t) = s(t) r_i(t) + (1 - s(t))/(N - 1) * sum_{j != i} r_j(t),

where s(t) is annealed from 1.0 (pure self-interest) towards 1/N
over training. Logging is kept consistent with IPPO/SVO/MAPPO.
"""
import sys
sys.path.append('/home/shuqing/SocialJax')
import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
from gymnax.wrappers.purerl import LogWrapper as GymnaxLogWrapper, FlattenObservationWrapper
import socialjax
from socialjax.wrappers.baselines import LogWrapper
import hydra
from omegaconf import OmegaConf
import wandb
import copy
import pickle
import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from algorithms.utils.cleanup_logging import init_wandb_run, log_eval_episode


class CNN(nn.Module):
    activation: str = "relu"
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))  # Flatten

        x = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(x)
        x = activation(x)

        return x


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "relu"
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embedding = CNN(self.activation, dtype=self.dtype)(x)

        actor_mean = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(embedding)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean.astype(jnp.float32))

        critic = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
            dtype=self.dtype,
            param_dtype=jnp.float32,
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic.astype(jnp.float32), axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[:, a] for a in agent_list])
    return x.reshape((num_actors, -1))


def batchify_dict(x: dict, agent_list, num_actors):
    x = jnp.stack([x[str(a)] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    # Ensure at least one update even if TOTAL_TIMESTEPS is small (e.g., smoke tests).
    config["NUM_UPDATES"] = max(
        1, config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    env = LogWrapper(env, replace_info=False)

    # Reward-exchange schedule: s starts at 1.0 and anneals towards 1/N over training.
    num_agents = config["ENV_KWARGS"]["num_agents"]
    s_start = 1.0
    s_end = 1.0 / num_agents
    s_anneal = optax.linear_schedule(
        init_value=s_start,
        end_value=s_end,
        transition_steps=config["NUM_UPDATES"],
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        network = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])

        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))

        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(
                    -1, *(env.observation_space()[0]).shape
                )

                pi, value = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )

                env_act = [v for v in env_act.values()]

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                # Apply reward-exchange mechanism to the per-agent rewards.
                # reward is a dict: agent_id -> (NUM_ENVS,) array.
                # Build (NUM_ENVS, num_agents) matrix of raw rewards.
                rewards_mat = jnp.stack(
                    [reward[a] for a in env.agents], axis=1
                )  # (NUM_ENVS, N)
                total_rewards = rewards_mat.sum(axis=1, keepdims=True)
                rewards_others = total_rewards - rewards_mat

                s_val = s_anneal(update_step)
                s = jnp.clip(s_val, 0.0, 1.0)
                other_coeff = jnp.where(
                    num_agents > 1,
                    (1.0 - s) / (num_agents - 1),
                    0.0,
                )
                exchanged_rewards = s * rewards_mat + other_coeff * rewards_others

                def _reshape_info(x):
                    size = x.size
                    num_actors = config["NUM_ACTORS"]
                    num_envs = config["NUM_ENVS"]
                    num_agents_local = config["ENV_KWARGS"]["num_agents"]
                    if size == num_actors:
                        return x.reshape((num_actors,))
                    if size == num_envs * num_agents_local:
                        return x.reshape((num_actors,))
                    if size == num_envs:
                        return jnp.repeat(x, num_agents_local)
                    if size == 1:
                        return jnp.repeat(x, num_actors)
                    return x

                info = jax.tree_map(_reshape_info, info)
                # Use exchanged rewards for learning (PPO/GAE), keeping original
                # rewards in `info["original_rewards"]` via the wrapper.
                rew_batch = exchanged_rewards.T.reshape((config["NUM_ACTORS"], -1))
                transition = Transition(
                    batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action,
                    value,
                    rew_batch.squeeze(),
                    log_prob,
                    obs_batch,
                    info,
                )
                runner_state = (train_state, env_state, obsv, update_step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, update_step, rng = runner_state
            last_obs_batch = jnp.stack(
                [last_obs[:, a, ...] for a in env.agents]
            ).reshape(-1, *(env.observation_space()[0]).shape)
            _, last_val = network.apply(train_state.params, last_obs_batch)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )

                    reward_mean = jnp.mean(reward, axis=0)
                    reward = reward - reward_mean

                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )

                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )

                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                train_state, traj_batch, advantages, targets, rng = update_state

                def loss_fn(params, traj_batch, gae, targets, rng):
                    rng, _rng = jax.random.split(rng)
                    batch_size = traj_batch.obs.shape[0]
                    idx = jax.random.permutation(_rng, batch_size)
                    idx = idx[: config["MINIBATCH_SIZE"]]

                    b_obs = traj_batch.obs[idx]
                    b_action = traj_batch.action[idx]
                    b_log_prob = traj_batch.log_prob[idx]
                    b_advantages = gae[idx]
                    b_targets = targets[idx]

                    pi, value = network.apply(params, b_obs)
                    log_prob = pi.log_prob(b_action)
                    entropy = pi.entropy().mean()

                    ratio = jnp.exp(log_prob - b_log_prob)
                    pg_loss1 = -b_advantages * ratio
                    pg_loss2 = -b_advantages * jnp.clip(
                        ratio,
                        1.0 - config["CLIP_EPS"],
                        1.0 + config["CLIP_EPS"],
                    )
                    pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

                    value_loss = ((b_targets - value) ** 2).mean()
                    loss = (
                        pg_loss
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                    )
                    return loss, {
                        "pg_loss": pg_loss,
                        "v_loss": value_loss,
                        "entropy": entropy,
                    }

                grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
                (loss, loss_info), grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets, rng
                )
                train_state = train_state.apply_gradients(grads=grads)
                return (train_state, traj_batch, advantages, targets, rng), loss_info

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)

            update_step = update_step + 1
            # Compute apple/clean counts before averaging, same as IPPO.
            apples_per_actor = clean_per_actor = clean_rate = None
            if ("clean_action_info" in metric) and (
                "apples_collected_per_agent" in metric or "original_rewards" in metric
            ):
                num_actors = config["NUM_ACTORS"]
                num_agents_local = config["ENV_KWARGS"]["num_agents"]
                apples_source = (
                    metric["apples_collected_per_agent"]
                    if "apples_collected_per_agent" in metric
                    else metric["original_rewards"] / num_agents_local
                )
                if apples_source.size and apples_source.size % num_actors == 0:
                    apples_flat = apples_source.reshape(-1, num_actors)
                    clean_flat = metric["clean_action_info"].reshape(-1, num_actors)
                    apples_per_actor = apples_flat.sum(axis=0)
                    clean_per_actor = clean_flat.sum(axis=0)

                    def gini(x):
                        x = jnp.sort(x)
                        n = x.size
                        total = jnp.sum(x)
                        return jax.lax.cond(
                            total <= 0,
                            lambda: 0.0,
                            lambda: (2 * jnp.sum((jnp.arange(1, n + 1) * x))
                                     / (n * total) - (n + 1) / n),
                        )

                    clean_rate = clean_flat.sum() / (
                        config["NUM_STEPS"] * num_actors
                    )

            metric = jax.tree_map(lambda x: x.mean(), metric)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["clean_action_info"] = (
                metric["clean_action_info"] * config["ENV_KWARGS"]["num_inner_steps"]
            )
            if apples_per_actor is not None:
                metric["train/apples_total"] = apples_per_actor.sum()
                metric["train/apples_per_actor"] = apples_per_actor
                metric["train/clean_actions_per_actor"] = clean_per_actor
                metric["train/clean_action_rate"] = clean_rate
                metric["train/returns_gini"] = gini(apples_per_actor)

            jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, update_step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, 0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def single_run(config):
    config = OmegaConf.to_container(config)
    run_name = config.get("RUN_NAME") or f'{config.get("CONDITION", "C4")}_seed{config.get("SEED", 0)}'
    init_wandb_run(
        project=config.get("PROJECT", "socialjax-cleanup"),
        entity=config.get("ENTITY") or None,
        run_name=run_name,
        condition=config.get("CONDITION", "C4"),
        algo_name=config.get("ALGO_NAME", "PPO_RE"),
        mechanism_class=config.get("MECHANISM_CLASS", "ClassIV"),
        seed=config.get("SEED", 0),
        total_timesteps=int(config.get("TOTAL_TIMESTEPS", 0)),
        extra_config={"env_name": config.get("ENV_NAME", "clean_up")},
    )
    tags = ["PPO_RE"] + (config.get("WANDB_TAGS") or [])
    if wandb.run is not None:
        wandb.run.tags = tuple(tags)

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)
    _ = out


@hydra.main(version_base=None, config_path="config", config_name="ppo_re_cnn_cleanup")
def main(config):
    if config["TUNE"]:
        raise NotImplementedError("Tuning for PPO_RE is not implemented.")
    else:
        single_run(config)


if __name__ == "__main__":
    main()
