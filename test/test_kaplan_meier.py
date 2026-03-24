import jax
import jax.numpy as jnp
from kairos.kaplan_meier import kaplan_meier_estimator


def test_kaplan_merier_single_time():
    times = jnp.array([1, 1, 1])
    events = jnp.array([1, 0, 0])
    expected_probs = jnp.array([1 - 1 / 3, 1 - 1 / 3])

    probs = kaplan_meier_estimator(times, events, 2)
    print(probs, expected_probs)
    assert jnp.allclose(probs, expected_probs, atol=1e-6)


def test_kaplan_meier_no_censor():
    times = jnp.array([1, 2, 2, 3])
    events = jnp.array([1, 1, 1, 1])
    expected_probs = jnp.cumprod(
        jnp.array([1 - 1 / 4, 1 - 2 / 3, 1 - 1 / 1, 1 - 0 / 1, 1 - 0 / 1])
    )

    probs = kaplan_meier_estimator(times, events, 5)
    print(probs, expected_probs)
    assert jnp.allclose(probs, expected_probs, atol=1e-6)


def test_kaplan_meier_only_censor():
    times = jnp.array([1, 2, 2, 3])
    events = jnp.array([0, 0, 0, 0])
    expected_probs = jnp.cumprod(jnp.array([1 - 0, 1 - 0, 1 - 0, 1 - 0, 1 - 0]))

    probs = kaplan_meier_estimator(times, events, 5)
    print(probs, expected_probs)
    assert jnp.allclose(probs, expected_probs, atol=1e-6)


def test_kaplan_meier():
    times = jnp.array([1, 2, 2, 3])
    events = jnp.array([1, 0, 0, 1])
    expected_probs = jnp.cumprod(
        jnp.array([1 - 1 / 4, 1 - 0, 1 - 1 / 1, 1 - 0 / 1, 1 - 0 / 1])
    )

    probs = kaplan_meier_estimator(times, events, 5)
    print(probs, expected_probs)
    assert jnp.allclose(probs, expected_probs, atol=1e-6)


def test_survival_is_monotone_decreasing():
    key = jax.random.PRNGKey(0)
    times = jax.random.randint(key, (100,), 1, 50).astype(jnp.float32)
    events = jax.random.bernoulli(key, 0.7, (100,)).astype(jnp.int32)
    probs = kaplan_meier_estimator(times, events, max_unique=50)
    assert jnp.all(jnp.diff(probs) <= 0)


def test_survival_bounded():
    key = jax.random.PRNGKey(0)
    times = jax.random.randint(key, (100,), 1, 50).astype(jnp.float32)
    events = jax.random.bernoulli(key, 0.7, (100,)).astype(jnp.int32)
    probs = kaplan_meier_estimator(times, events, max_unique=50)
    assert jnp.all(probs >= 0) and jnp.all(probs <= 1)
