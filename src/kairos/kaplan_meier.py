import jax.numpy as jnp


def kaplan_meier_estimator(
    times: jnp.ndarray,
    events: jnp.ndarray,
    max_unique: int,
) -> jnp.ndarray:
    """_summary_

    Args:
        times (jnp.ndarray): (n,) array of times
        events (jnp.ndarray): (n,) array of events
        max_unique (int): maximum number of unique times

    Returns:
        jnp.ndarray: (max_unique, ) survival probabilities
    """

    unique_times = jnp.unique(jnp.sort(times), size=max_unique, fill_value=-1)

    at_risk = jnp.sum((unique_times[:, None] <= times[None, :]), axis=1)
    event_at_t = jnp.sum(
        (unique_times[:, None] == times[None, :]) * (events[None, :] == 1), axis=1
    )

    probs = jnp.cumprod(1 - (event_at_t / at_risk))
    return probs
