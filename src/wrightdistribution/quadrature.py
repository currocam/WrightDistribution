"""Gauss-Jacobi quadrature via the Golub-Welsch algorithm in JAX."""

import jax.numpy as jnp
from jax.scipy.special import betaln


def gauss_jacobi(n: int, alpha, beta):
    """Compute n-point Gauss-Jacobi quadrature nodes and weights on [-1, 1].

    Uses the Golub-Welsch algorithm: builds the symmetric tridiagonal Jacobi
    matrix from the three-term recurrence coefficients for Jacobi polynomials,
    then computes nodes (eigenvalues) and weights (from eigenvectors).

    Weight function: w(x) = (1-x)^alpha * (1+x)^beta on [-1, 1].

    Args:
        n: Number of quadrature nodes (static int, not traced by JAX).
        alpha: Jacobi parameter (> -1), exponent on (1-x).
        beta: Jacobi parameter (> -1), exponent on (1+x).

    Returns:
        (nodes, weights): arrays of shape (n,).
    """
    alpha = jnp.asarray(alpha, dtype=jnp.float64)
    beta = jnp.asarray(beta, dtype=jnp.float64)

    # Recurrence coefficients for monic Jacobi polynomials.
    # Three-term recurrence: P_{k+1}(x) = (x - a_k) P_k(x) - b_k P_{k-1}(x)
    i = jnp.arange(n, dtype=jnp.float64)
    ab = alpha + beta

    # Diagonal: a_i = (beta^2 - alpha^2) / ((2i + ab)(2i + ab + 2))
    # Special case i=0: a_0 = (beta - alpha) / (ab + 2)
    denom = (2 * i + ab) * (2 * i + ab + 2)
    diag = jnp.where(
        denom == 0,
        0.0,
        (beta**2 - alpha**2) / denom,
    )

    # Off-diagonal (for i = 1, ..., n-1):
    # b_i = 4*i*(i+alpha)*(i+beta)*(i+ab) / ((2i+ab)^2 * ((2i+ab)^2 - 1))
    j = jnp.arange(1, n, dtype=jnp.float64)
    numer = 4 * j * (j + alpha) * (j + beta) * (j + ab)
    s = 2 * j + ab
    denom_off = s**2 * (s**2 - 1)
    b = numer / denom_off
    off_diag = jnp.sqrt(b)

    # Build full symmetric tridiagonal matrix and diagonalize.
    J = jnp.diag(diag) + jnp.diag(off_diag, 1) + jnp.diag(off_diag, -1)
    nodes, vecs = jnp.linalg.eigh(J)

    # Weights: w_i = mu_0 * v_{i,0}^2
    # where mu_0 = integral of (1-x)^alpha * (1+x)^beta over [-1,1]
    #            = 2^(alpha+beta+1) * B(alpha+1, beta+1)
    log_mu0 = (ab + 1) * jnp.log(2.0) + betaln(alpha + 1, beta + 1)
    mu0 = jnp.exp(log_mu0)
    weights = mu0 * vecs[0, :] ** 2

    return nodes, weights
