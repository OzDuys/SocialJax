"""Reciprocity condition (C4) for CleanUp.

Implements innovator + imitator training with intrinsic reciprocity reward.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any, NamedTuple, Sequence

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import socialjax
import wandb
from socialjax.wrappers.baselines import LogWrapper
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf
from PIL import Image

from algorithms.utils.cleanup_logging import init_wandb_run


class CNN(nn.Module):
    activation: str = "relu"
    dtype: Any = jnp.bfloat16

    @nn.compact
    def __call__(self, x):
        x = x.astype(self.dtype)
        activation = nn.relu if self.activation == "relu" else nn.tanh

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
        x = x.reshape((x.shape[0], -1))
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
        activation = nn.relu if self.activation == "relu" else nn.tanh

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
        critic = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic)

        return pi, jnp.squeeze(critic.astype(jnp.float32), axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    clean_action: jnp.ndarray
    env_reward: jnp.ndarray


class PPOBatch(NamedTuple):
    transition: Transition
    advantages: jnp.ndarray
    targets: jnp.ndarray


def _compute_gae(traj_batch: Transition, last_val: jnp.ndarray, gamma: float, gae_lambda: float):
    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = transition.done, transition.value, transition.reward
        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    targets = advantages + traj_batch.value
    return advantages, targets


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    env = LogWrapper(env, replace_info=False)
    num_agents = env.num_agents
    innovator_idx = int(config.get("INNOVATOR_INDEX", 0))
    imitator_indices = [i for i in range(num_agents) if i != innovator_idx]
    num_imitators = len(imitator_indices)
    num_actors_total = config["NUM_ENVS"] * num_agents
    config["NUM_ACTORS"] = num_actors_total

    if config["PARAMETER_SHARING"]:
        num_actors_inv = config["NUM_ENVS"]
        num_actors_im = config["NUM_ENVS"] * num_imitators
    else:
        raise ValueError("C4 requires shared imitator parameters; set PARAMETER_SHARING=True.")

    num_updates = max(
        1, config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / num_updates
        )
        return config["LR"] * frac

    def train(rng):
        # Init networks
        network_inv = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        network_im = ActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        rng, rng_inv, rng_im = jax.random.split(rng, 3)
        init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))
        params_inv = network_inv.init(rng_inv, init_x)
        params_im = network_im.init(rng_im, init_x)

        tx_fn = optax.adam(learning_rate=linear_schedule, eps=1e-5) if config["ANNEAL_LR"] else optax.adam(config["LR"], eps=1e-5)
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), tx_fn)
        train_state_inv = TrainState.create(apply_fn=network_inv.apply, params=params_inv, tx=tx)
        train_state_im = TrainState.create(apply_fn=network_im.apply, params=params_im, tx=tx)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        niceness = jnp.zeros((config["NUM_ENVS"], num_agents))

        def _reshape_info(x):
            size = x.size
            if size == num_actors_total:
                return x.reshape((num_actors_total,))
            if size == config["NUM_ENVS"] * num_agents:
                return x.reshape((num_actors_total,))
            if size == config["NUM_ENVS"]:
                return jnp.repeat(x, num_agents)
            if size == 1:
                return jnp.repeat(x, num_actors_total)
            return x

        def _env_step(runner_state, unused):
            ts_inv, ts_im, env_state, last_obs, niceness, rng = runner_state
            rng, rng_inv, rng_im, rng_step = jax.random.split(rng, 4)

            obs_by_agent = jnp.transpose(last_obs, (1, 0, 2, 3, 4))  # (agents, envs, ...)
            innov_obs = obs_by_agent[innovator_idx]
            imit_obs = jnp.concatenate([obs_by_agent[i] for i in imitator_indices], axis=0)

            pi_inv, value_inv = network_inv.apply(ts_inv.params, innov_obs)
            action_inv = pi_inv.sample(seed=rng_inv)
            logp_inv = pi_inv.log_prob(action_inv)

            pi_im, value_im = network_im.apply(ts_im.params, imit_obs)
            action_im = pi_im.sample(seed=rng_im)
            logp_im = pi_im.log_prob(action_im)

            action_im = action_im.reshape((num_imitators, config["NUM_ENVS"]))
            logp_im = logp_im.reshape((num_imitators, config["NUM_ENVS"]))
            value_im = value_im.reshape((num_imitators, config["NUM_ENVS"]))

            env_act = []
            im_ptr = 0
            for idx in range(num_agents):
                if idx == innovator_idx:
                    env_act.append(action_inv)
                else:
                    env_act.append(action_im[im_ptr])
                    im_ptr += 1

            rng_env = jax.random.split(rng_step, config["NUM_ENVS"])
            obsv, env_state, reward, done, info = jax.vmap(env.step, in_axes=(0, 0, 0))(rng_env, env_state, env_act)

            done_flag = done["__all__"] if isinstance(done, dict) else done
            done_float = done_flag.astype(jnp.float32)
            clean_indicator = info.get("clean_action_info", jnp.zeros_like(reward))
            clean_indicator = jnp.asarray(clean_indicator)
            niceness_next = (config["NICENESS_DECAY"] * niceness + clean_indicator) * (1.0 - done_float[:, None])

            niceness_inv = niceness_next[:, innovator_idx]
            niceness_im = niceness_next[:, imitator_indices]
            intrinsic = -((niceness_im - niceness_inv[:, None]) ** 2)

            reward_im = reward[:, imitator_indices] + config["IMITATOR_LAMBDA"] * intrinsic
            reward_inv = reward[:, innovator_idx]
            info_flat = jax.tree_map(_reshape_info, info)

            # Flatten imitator trajectories for PPO update
            transition_inv = Transition(
                done=done_float,
                action=action_inv,
                value=value_inv,
                reward=reward_inv,
                log_prob=logp_inv,
                obs=innov_obs,
                clean_action=clean_indicator[:, innovator_idx],
                env_reward=reward_inv,
            )
            transition_im = Transition(
                done=jnp.repeat(done_float[None, :], num_imitators, axis=0).reshape(-1,),
                action=action_im.reshape(-1),
                value=value_im.reshape(-1),
                reward=reward_im.reshape(-1),
                log_prob=logp_im.reshape(-1),
                obs=imit_obs,
                clean_action=clean_indicator[:, imitator_indices].reshape(-1),
                env_reward=reward[:, imitator_indices].reshape(-1),
            )

            runner_state = (ts_inv, ts_im, env_state, obsv, niceness_next, rng)
            return runner_state, (transition_inv, transition_im, niceness_next, info_flat)

        def _calculate_advantages(traj_inv, traj_im, last_obs, ts_inv, ts_im):
            obs_by_agent = jnp.transpose(last_obs, (1, 0, 2, 3, 4))
            innov_obs = obs_by_agent[innovator_idx]
            imit_obs = jnp.concatenate([obs_by_agent[i] for i in imitator_indices], axis=0)

            _, last_val_inv = network_inv.apply(ts_inv.params, innov_obs)
            _, last_val_im = network_im.apply(ts_im.params, imit_obs)
            last_val_im = last_val_im.reshape((num_imitators * config["NUM_ENVS"],))

            adv_inv, targets_inv = _compute_gae(traj_inv, last_val_inv, config["GAMMA"], config["GAE_LAMBDA"])
            adv_im, targets_im = _compute_gae(traj_im, last_val_im, config["GAMMA"], config["GAE_LAMBDA"])
            return adv_inv, targets_inv, adv_im, targets_im

        def _update_epoch(train_state, minibatches, network):
            def _update_minibatch(state, batch_info):
                trans, adv, target = batch_info
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                def loss_fn(params):
                    pi, value = network.apply(params, trans.obs)
                    log_prob = pi.log_prob(trans.action)
                    ratio = jnp.exp(log_prob - trans.log_prob)
                    surr1 = ratio * adv
                    surr2 = jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"]) * adv
                    entropy = pi.entropy().mean()
                    v_loss = 0.5 * ((target - value) ** 2).mean()
                    loss = -jnp.minimum(surr1, surr2).mean() + config["VF_COEF"] * v_loss - config["ENT_COEF"] * entropy
                    return loss, (v_loss, entropy)

                (loss, (v_loss, entropy)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
                state = state.apply_gradients(grads=grads)
                metrics = {
                    "loss": loss,
                    "value_loss": v_loss,
                    "entropy": entropy,
                }
                return state, metrics

            return jax.lax.scan(_update_minibatch, train_state, minibatches)

        def _update_step(runner_state, unused):
            ts_inv, ts_im, env_state, obsv, update_step, rng, niceness = runner_state
            runner_state_inner = (ts_inv, ts_im, env_state, obsv, niceness, rng)
            runner_state_inner, traj = jax.lax.scan(_env_step, runner_state_inner, None, config["NUM_STEPS"])
            (ts_inv, ts_im, env_state, last_obs, niceness, rng) = runner_state_inner
            traj_inv, traj_im, niceness_seq, info_seq = traj

            adv_inv, targets_inv, adv_im, targets_im = _calculate_advantages(traj_inv, traj_im, last_obs, ts_inv, ts_im)

            batch_inv = PPOBatch(
                transition=Transition(
                    done=traj_inv.done.reshape(-1),
                    action=traj_inv.action.reshape(-1),
                    value=traj_inv.value.reshape(-1),
                    reward=traj_inv.reward.reshape(-1),
                    log_prob=traj_inv.log_prob.reshape(-1),
                    obs=traj_inv.obs.reshape((config["NUM_STEPS"] * num_actors_inv, ) + traj_inv.obs.shape[2:]),
                    clean_action=traj_inv.clean_action.reshape(-1),
                    env_reward=traj_inv.env_reward.reshape(-1),
                ),
                advantages=adv_inv.reshape(-1),
                targets=targets_inv.reshape(-1),
            )

            batch_im = PPOBatch(
                transition=Transition(
                    done=traj_im.done.reshape(-1),
                    action=traj_im.action.reshape(-1),
                    value=traj_im.value.reshape(-1),
                    reward=traj_im.reward.reshape(-1),
                    log_prob=traj_im.log_prob.reshape(-1),
                    obs=traj_im.obs.reshape((config["NUM_STEPS"] * num_actors_im, ) + traj_im.obs.shape[2:]),
                    clean_action=traj_im.clean_action.reshape(-1),
                    env_reward=traj_im.env_reward.reshape(-1),
                ),
                advantages=adv_im.reshape(-1),
                targets=targets_im.reshape(-1),
            )

            # Shuffle with fresh rngs
            rng, rng_inv, rng_im = jax.random.split(rng, 3)
            perm_inv = jax.random.permutation(rng_inv, batch_inv.transition.done.shape[0])
            perm_im = jax.random.permutation(rng_im, batch_im.transition.done.shape[0])
            shuffle_inv = jax.tree_map(lambda x: jnp.take(x, perm_inv, axis=0), batch_inv)
            shuffle_im = jax.tree_map(lambda x: jnp.take(x, perm_im, axis=0), batch_im)

            minibatches_inv = jax.tree_map(
                lambda x: jnp.reshape(
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffle_inv,
            )
            minibatches_im = jax.tree_map(
                lambda x: jnp.reshape(
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffle_im,
            )

            ts_inv, metrics_inv = _update_epoch(ts_inv, (minibatches_inv.transition, minibatches_inv.advantages, minibatches_inv.targets), network_inv)
            ts_im, metrics_im = _update_epoch(ts_im, (minibatches_im.transition, minibatches_im.advantages, minibatches_im.targets), network_im)

            def gini(x):
                x = jnp.sort(x)
                n = x.size
                total = jnp.sum(x)
                return jax.lax.cond(
                    total <= 0,
                    lambda: 0.0,
                    lambda: (2 * jnp.sum((jnp.arange(1, n + 1) * x)) / (n * total) - (n + 1) / n),
                )

            apples_per_actor = clean_per_actor = clean_rate = None
            if "apples_collected_per_agent" in info_seq or "original_rewards" in info_seq:
                apples_source = (
                    info_seq["apples_collected_per_agent"]
                    if "apples_collected_per_agent" in info_seq
                    else info_seq["original_rewards"] / num_agents
                )
                if apples_source.size and apples_source.size % num_actors_total == 0:
                    apples_flat = apples_source.reshape(-1, num_actors_total)
                    clean_flat = (
                        info_seq["clean_action_info"].reshape(-1, num_actors_total)
                        if "clean_action_info" in info_seq
                        else jnp.zeros_like(apples_flat)
                    )
                    apples_per_actor = apples_flat.sum(axis=0)
                    clean_per_actor = clean_flat.sum(axis=0)
                    clean_rate = clean_flat.sum() / (config["NUM_STEPS"] * num_actors_total)

            metric = jax.tree_map(lambda x: x.mean(), info_seq)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["train/innovator_return"] = traj_inv.reward.sum() / config["NUM_ENVS"]
            metric["train/imitator_return"] = traj_im.reward.sum() / (config["NUM_ENVS"] * num_imitators)
            if "clean_action_info" in metric:
                metric["clean_action_info"] = metric["clean_action_info"] * config["ENV_KWARGS"]["num_inner_steps"]
            if clean_rate is not None:
                metric["train/clean_action_rate"] = clean_rate
            if apples_per_actor is not None:
                metric["train/apples_total"] = apples_per_actor.sum()
                metric["train/apples_per_actor"] = apples_per_actor
                metric["train/clean_actions_per_actor"] = clean_per_actor
                metric["train/returns_gini"] = gini(apples_per_actor)
            jax.debug.callback(lambda m: wandb.log(m), metric)

            update_step = update_step + 1
            runner_state = (ts_inv, ts_im, env_state, last_obs, update_step, rng, niceness)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state_inv, train_state_im, env_state, obsv, 0, _rng, niceness)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None, num_updates)
        return {"runner_state": runner_state, "metrics": metric}

    return train


def save_params(train_state, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params = jax.tree_util.tree_map(lambda x: np.array(x), train_state.params)
    with open(save_path, "wb") as f:
        pickle.dump(params, f)


def load_params(load_path):
    with open(load_path, "rb") as f:
        params = pickle.load(f)
    return jax.tree_util.tree_map(lambda x: jnp.array(x), params)


def evaluate(params_inv, params_im, env, config):
    rng = jax.random.PRNGKey(0)
    rng, _rng = jax.random.split(rng)
    obs, state = env.reset(_rng)

    innovator_idx = int(config.get("INNOVATOR_INDEX", 0))
    imitator_indices = [i for i in range(env.num_agents) if i != innovator_idx]
    num_imitators = len(imitator_indices)
    obs_shape = env.observation_space()[0].shape

    episode_returns = np.zeros(len(env.agents), dtype=float)
    apples_total = np.zeros(len(env.agents), dtype=float)
    clean_actions = np.zeros(len(env.agents), dtype=float)
    niceness = np.zeros(len(env.agents), dtype=float)
    cleaned_total = 0.0

    pics = [env.render(state)]
    root_dir = "evaluation/cleanup"
    Path(root_dir + "/state_pics").mkdir(parents=True, exist_ok=True)

    network_inv = ActorCritic(action_dim=env.action_space().n, activation=config["ACTIVATION"])
    network_im = ActorCritic(action_dim=env.action_space().n, activation=config["ACTIVATION"])

    for _ in range(config["GIF_NUM_FRAMES"]):
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape((len(env.agents),) + obs_shape)
        innov_obs = obs_batch[innovator_idx : innovator_idx + 1]
        imit_obs = jnp.stack([obs_batch[i] for i in imitator_indices]).reshape((num_imitators,) + obs_shape)

        rng, rng_inv, rng_im = jax.random.split(rng, 3)
        pi_inv, _ = network_inv.apply(params_inv, innov_obs)
        action_inv = int(pi_inv.sample(seed=rng_inv).squeeze())

        pi_im, _ = network_im.apply(params_im, imit_obs)
        actions_im = pi_im.sample(seed=rng_im).reshape((num_imitators,))

        env_act = []
        im_ptr = 0
        for idx in range(len(env.agents)):
            if idx == innovator_idx:
                env_act.append(action_inv)
            else:
                env_act.append(int(actions_im[im_ptr]))
                im_ptr += 1

        rng, _rng = jax.random.split(rng)
        obs, state, reward, done, info = env.step(_rng, state, env_act)
        done_flag = done["__all__"] if isinstance(done, dict) else done

        reward_np = np.asarray(list(reward.values()), dtype=float) if isinstance(reward, dict) else np.asarray(reward, dtype=float)
        episode_returns = episode_returns + reward_np
        cleaned_total += float(np.mean(np.asarray(info.get("cleaned_water", 0.0))))

        if "apples_collected_per_agent" in info:
            apples_total += np.asarray(info["apples_collected_per_agent"], dtype=float)
        elif "original_rewards" in info:
            apples_total += np.asarray(info["original_rewards"], dtype=float) / len(env.agents)
        if "clean_action_info" in info:
            clean = np.asarray(info["clean_action_info"], dtype=float)
            clean_actions += clean
            niceness = config["NICENESS_DECAY"] * niceness + clean

        pics.append(env.render(state))

        if done_flag:
            break

    n_agents = len(env.agents)
    episode_len = len(pics)
    gif_name = f"{config.get('ALGO_NAME','IPPO-Reciprocity')}_{config.get('CONDITION','C4')}_seed-{config.get('SEED',0)}_{n_agents}-agents_frames-{episode_len}.gif"
    gif_path = f"{root_dir}/{gif_name}"
    pics_pil = [Image.fromarray(np.array(img)) for img in pics]
    pics_pil[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        optimize=False,
        append_images=pics_pil[1:],
        duration=200,
        loop=0,
    )

    def gini(x):
        x = np.asarray(x, dtype=float).flatten()
        if np.allclose(x, 0):
            return 0.0
        x = np.sort(x)
        n = x.size
        cum = np.cumsum(x)
        return float((2 * np.arange(1, n + 1) @ x - (n + 1) * x.sum()) / (n * x.sum() + 1e-8))

    clean_rate = float(clean_actions.sum() / (episode_len * n_agents))
    apples_per_agent_count = (apples_total / n_agents).tolist()
    apples_total_count = float(apples_total.sum() / n_agents)
    innov_clean = float(clean_actions[innovator_idx] / episode_len)
    imit_clean = float(np.mean(clean_actions[imitator_indices] / episode_len))
    niceness_gap = float(np.abs(niceness[innovator_idx] - np.mean(niceness[imitator_indices])))

    wandb.log(
        {
            "step": int(config.get("TOTAL_TIMESTEPS", 0)),
            "eval/episode": 0,
            "eval/team_return": float(episode_returns.sum()),
            "eval/mean_return": float(episode_returns.mean()),
            "eval/fairness_std": float(episode_returns.std()),
            "eval/cleaned_total": cleaned_total,
            "eval/returns_per_agent": episode_returns.tolist(),
            "eval/returns_gini": gini(episode_returns),
            "eval/clean_actions_per_agent": clean_actions.tolist(),
            "eval/clean_action_rate": clean_rate,
            "eval/apples_per_agent": apples_per_agent_count,
            "eval/apples_total": apples_total_count,
            "eval/team_apples": apples_total_count,
            "eval/episode_length": episode_len,
            "eval/innovator_cleaning_rate": innov_clean,
            "eval/imitator_cleaning_rate": imit_clean,
            "eval/niceness_gap": niceness_gap,
        }
    )
    if wandb.run is not None:
        wandb.log({"Episode GIF": wandb.Video(gif_path, caption="Evaluation Episode", format="gif")})


def single_run(config):
    config = OmegaConf.to_container(config)

    run_name = config.get("RUN_NAME") or f'{config.get("CONDITION", "C4")}_seed{config.get("SEED", 0)}'
    init_wandb_run(
        project=config.get("PROJECT", "socialjax-cleanup"),
        entity=config.get("ENTITY") or None,
        run_name=run_name,
        condition=config.get("CONDITION", "C4"),
        algo_name=config.get("ALGO_NAME", "IPPO-Reciprocity"),
        mechanism_class=config.get("MECHANISM_CLASS", "ClassIV"),
        seed=config.get("SEED", 0),
        total_timesteps=int(config.get("TOTAL_TIMESTEPS", 0)),
        extra_config={"env_name": config.get("ENV_NAME", "clean_up")},
    )
    tags = ["C4", "Reciprocity"] + (config.get("WANDB_TAGS") or [])
    if wandb.run is not None:
        wandb.run.tags = tuple(tags)

    rng = jax.random.PRNGKey(config["SEED"])
    rngs = jax.random.split(rng, config["NUM_SEEDS"])
    train_jit = jax.jit(make_train(config))
    out = jax.vmap(train_jit)(rngs)

    print("** Saving Results **")
    filename = f'{config["ENV_NAME"]}_seed{config["SEED"]}'
    runner_state = out["runner_state"][0]
    train_state_inv = runner_state[0]
    train_state_im = runner_state[1]
    save_dir = "./checkpoints/reciprocity"
    os.makedirs(save_dir, exist_ok=True)
    save_params(train_state_inv, f"{save_dir}/{filename}_innovator.pkl")
    save_params(train_state_im, f"{save_dir}/{filename}_imitator.pkl")
    params_inv = load_params(f"{save_dir}/{filename}_innovator.pkl")
    params_im = load_params(f"{save_dir}/{filename}_imitator.pkl")

    evaluate(params_inv, params_im, socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"]), config)


@hydra.main(version_base=None, config_path="config", config_name="ippo_cnn_cleanup_c4")
def main(config):
    single_run(config)


if __name__ == "__main__":
    main()
