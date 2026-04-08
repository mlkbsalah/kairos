import jax.numpy as jnp

def cox_efron_log_likelihood(
    beta: jnp.ndarray,          
    X: jnp.ndarray,             
    risk_matrix: jnp.ndarray,   
    death_matrix: jnp.ndarray,  
    d_k: jnp.ndarray,           
    valid_mask: jnp.ndarray,    
    max_j: int,
):
    theta = jnp.exp(X @ beta)                       
    Xb = X @ beta                          

    numerator = death_matrix @ Xb          

    sum_risk_set = risk_matrix @ theta
    sum_tied_set = death_matrix @ theta

    j_val = jnp.arange(0, max_j)

    denom = (
        sum_risk_set[:, None]
        - (j_val[None, :] / d_k[:, None]) * sum_tied_set[:, None]
    )                                                

    j_valid_mask = j_val[None, :] < d_k[:, None]   
    safe_denom = jnp.where(j_valid_mask, denom, 1.0)
    log_denom = jnp.where(j_valid_mask, jnp.log(safe_denom), 0.0)

    log_denom_sum = jnp.sum(log_denom, axis=1)     

    ll = jnp.sum(valid_mask * (numerator - log_denom_sum))

    return ll


