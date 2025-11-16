import numpy as np
import wandb


def init_wandb_run(
    *,
    project: str,
    entity: str | None,
    run_name: str,
    condition: str,
    algo_name: str,
    mechanism_class: str,
    seed: int,
    total_timesteps: int,
    extra_config: dict | None = None,
):
    """
    Initialize a W&B run with shared metadata for the CleanUp experiments.
    """
    config = {
        "condition": condition,
        "algo": algo_name,
        "mechanism_class": mechanism_class,
        "seed": seed,
        "total_timesteps": total_timesteps,
    }
    if extra_config:
        config.update(extra_config)

    wandb.init(
        project=project,
        entity=entity or None,
        name=run_name,
        config=config,
    )


def log_eval_episode(step: int, episode_idx: int, returns, cleaned_total: float):
    """
    Log evaluation metrics for a single episode.
    """
    returns = np.asarray(returns, dtype=float)
    team_return = float(returns.sum())
    mean_return = float(returns.mean())
    fairness_std = float(returns.std())

    wandb.log(
        {
            "step": step,
            "eval/episode": episode_idx,
            "eval/team_return": team_return,
            "eval/mean_return": mean_return,
            "eval/fairness_std": fairness_std,
            "eval/cleaned_total": float(cleaned_total),
        }
    )
