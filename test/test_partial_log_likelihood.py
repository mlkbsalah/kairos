import jax
import numpy as np
import jax.numpy as jnp
from kairos.partial_log_likelihood import partial_negloglik


def make_cox_data(n, beta_true, seed=199111):
    rng = np.random.default_rng(seed)

    beta_true = np.asarray(beta_true)
    p = beta_true.shape[0]

    X = rng.normal(0, 1, (n, p))
    rate = np.exp(X @ beta_true)
    T = rng.exponential(1.0 / rate)
    C = rng.exponential(2.0, n)

    times = np.minimum(T, C)
    events = (T <= C).astype(np.float32)

    # add ties
    tie_indices = rng.choice(n, size=3, replace=False)
    times[tie_indices] = times[tie_indices[0]]
    events[tie_indices] = 1.0

    return (
        jnp.array(X, dtype=jnp.float32),
        jnp.array(times, dtype=jnp.float32),
        jnp.array(events, dtype=jnp.float32),
    )


def _build_matrices(X, times, events, max_unique):
    unique_times = jnp.sort(jnp.unique(times[events == 1]))

    risk_matrix = (unique_times[:, None] <= times[None, :]).astype(jnp.float32)
    death_matrix = ((unique_times[:, None] == times[None, :]) * events[None, :]).astype(
        jnp.float32
    )

    d_k = jnp.sum(death_matrix, axis=1)
    max_j = int(jnp.max(d_k))

    return risk_matrix, death_matrix, d_k, max_j


def test_partial_negloglik():
    X, times, events = make_cox_data(10, [0.5, -0.3])
    risk_matrix, death_matrix, d_k, max_j = _build_matrices(
        X, times, events, max_unique=100
    )

    beta = jnp.array([0.5, -0.3], dtype=jnp.float32)
    ll = partial_negloglik(beta, X, risk_matrix, death_matrix, d_k, max_j)

    assert jnp.isfinite(ll), "Partial log-likelihood should be finite"


def test_partial_negloglik_grad():
    X, times, events = make_cox_data(100, [0.5, -0.3])
    risk_matrix, death_matrix, d_k, max_j = _build_matrices(
        X, times, events, max_unique=100
    )

    beta = jnp.array([0.5, -0.3], dtype=jnp.float32)
    grad_fn = jax.grad(partial_negloglik, argnums=0)
    grad = grad_fn(beta, X, risk_matrix, death_matrix, d_k, max_j)
    assert jnp.isfinite(grad).all(), "Gradient should be finite"
    assert grad.shape == beta.shape, "Gradient should have the same shape as beta"


def test_partial_negloglik_hessian():
    X, times, events = make_cox_data(100, [0.5, -0.3])
    risk_matrix, death_matrix, d_k, max_j = _build_matrices(
        X, times, events, max_unique=100
    )

    beta = jnp.array([0.5, -0.3], dtype=jnp.float32)
    hessian_fn = jax.hessian(partial_negloglik, argnums=0)
    hessian = hessian_fn(beta, X, risk_matrix, death_matrix, d_k, max_j)

    assert jnp.isfinite(hessian).all(), "Hessian should be finite"
    assert hessian.shape == (beta.shape[0], beta.shape[0]), (
        "Hessian should be square matrix of shape (p, p)"
    )
