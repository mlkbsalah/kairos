import jax.numpy as jnp
from typing import Tuple
def nelson_aalen_estimator(
    times: jnp.ndarray,
    events: jnp.ndarray,
    max_unique: int,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Estimate the cumulative hazard function using the Nelson-Aalen estimator.

    Args:
        times (jnp.ndarray): (n,) array of times
        events (jnp.ndarray): (n,) array of events
        max_unique (int): maximum number of unique times

    Returns:
        jnp.ndarray: (max_unique, ) cumulative hazard rates
        jnp.ndarray: (max_unique, ) unique times
    """

    unique_times = jnp.unique(jnp.sort(times), size=max_unique, fill_value=-1)

    at_risk = jnp.sum((unique_times[:, None] <= times[None, :]), axis=1)
    events_at_t = jnp.sum(
        (unique_times[:, None] == times[None, :]) * (events[None, :] == 1), axis=1
    )

    probs = jnp.cumsum(events_at_t / at_risk)
    return probs, unique_times
