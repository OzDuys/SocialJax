from socialjax.environments import (
    # Social dilemma environments
    Clean_up,
)

# Registry of all available environments
REGISTERED_ENVS = [
    # Social dilemma environments
    "clean_up",
]


def make(env_id: str, **env_kwargs):
    """A JAX-version of OpenAI's env.make(env_name), built off Gymnax"""
    if env_id not in REGISTERED_ENVS:
        raise ValueError(f"{env_id} is not in registered SocialJax environments")

    elif env_id == "clean_up":
        env = Clean_up(**env_kwargs)
    return env