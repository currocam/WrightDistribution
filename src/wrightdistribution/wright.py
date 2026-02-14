"""Wright distribution for population genetics, implemented in JAX.

The Wright distribution describes the stationary allele frequency distribution
under selection, mutation, and genetic drift. Its unnormalized PDF is:

    f(p) ∝ p^(2Nα - 1) · (1-p)^(2Nβ - 1) · exp(-Ns·p·(2h + (1-2h)·(2-p)))

where p is the allele frequency on [0, 1].
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .quadrature import gauss_jacobi

N_NODES = 64


class Wright(NamedTuple):
    """Wright distribution parameters.

    Attributes:
        Ns: Scaled selection coefficient (Nₑ·s).
        Na: Scaled mutation rate α (Nₑ·α, not 2Nₑ·α).
        Nb: Scaled mutation rate β (Nₑ·β, not 2Nₑ·β).
        h: Dominance coefficient.
        Z: Normalization constant (precomputed).
        n_nodes: Number of Gauss-Jacobi quadrature nodes.
    """

    Ns: jax.Array
    Na: jax.Array
    Nb: jax.Array
    h: jax.Array
    Z: jax.Array
    n_nodes: int


def wright(Ns, Na, Nb, h, n_nodes=N_NODES):
    """Construct a Wright distribution.

    Args:
        Ns: Scaled selection coefficient (Nₑ·s).
        Na: Scaled mutation rate for allele p (Nₑ·α).
        Nb: Scaled mutation rate for allele q=1-p (Nₑ·β).
        h: Dominance coefficient (0 ≤ h ≤ 1).
        n_nodes: Number of quadrature nodes for integration.

    Returns:
        Wright NamedTuple with precomputed normalization constant.
    """
    Ns = jnp.asarray(Ns, dtype=jnp.float64)
    Na = jnp.asarray(Na, dtype=jnp.float64)
    Nb = jnp.asarray(Nb, dtype=jnp.float64)
    h = jnp.asarray(h, dtype=jnp.float64)
    Z = zfun(Ns, 2 * Na, 2 * Nb, h, n_nodes)
    return Wright(Ns=Ns, Na=Na, Nb=Nb, h=h, Z=Z, n_nodes=n_nodes)


def cfun(p, Ns, h):
    """Exponential selection factor: exp(-Ns·p·(2h + (1-2h)·(2-p)))."""
    return jnp.exp(log_cfun(p, Ns, h))


def log_cfun(p, Ns, h):
    """Log of the selection factor: -Ns·p·(2h + (1-2h)·(2-p))."""
    return -Ns * p * (2 * h + (1 - 2 * h) * (2 - p))


def zfun(Ns, A, B, h, n_nodes=N_NODES):
    """Normalization constant via Gauss-Jacobi quadrature.

    Computes ∫₀¹ p^(A-1)·(1-p)^(B-1)·C(p) dp where A=2Nα, B=2Nβ.
    The Jacobi weight (1-x)^(B-1)·(1+x)^(A-1) handles the boundary singularity.
    """
    # Jacobi parameters: weight is (1-x)^alpha * (1+x)^beta on [-1,1]
    # We need (1-x)^(B-1) * (1+x)^(A-1) to match p^(A-1)*(1-p)^(B-1)
    nodes, weights = gauss_jacobi(n_nodes, B - 1, A - 1)
    p_nodes = (nodes + 1) / 2
    values = cfun(p_nodes, Ns, h)
    return jnp.dot(weights, values) / 2 ** (A + B - 1)


def yfun(Ns, A, B, h, n_nodes=N_NODES):
    """Numerator for E[p] via Gauss-Jacobi quadrature.

    Computes ∫₀¹ p·p^(A-1)·(1-p)^(B-1)·C(p) dp.
    """
    nodes, weights = gauss_jacobi(n_nodes, B - 1, A - 1)
    p_nodes = (nodes + 1) / 2
    values = p_nodes * cfun(p_nodes, Ns, h)
    return jnp.dot(weights, values) / 2 ** (A + B - 1)


def wright_pdf(d: Wright, p):
    """Probability density function of the Wright distribution at p."""
    p = jnp.asarray(p, dtype=jnp.float64)
    A, B = 2 * d.Na, 2 * d.Nb
    return p ** (A - 1) * (1 - p) ** (B - 1) * cfun(p, d.Ns, d.h) / d.Z


def wright_logpdf(d: Wright, p):
    """Log probability density function of the Wright distribution at p."""
    p = jnp.asarray(p, dtype=jnp.float64)
    A, B = 2 * d.Na, 2 * d.Nb
    return (
        (A - 1) * jnp.log(p)
        + (B - 1) * jnp.log(1 - p)
        + log_cfun(p, d.Ns, d.h)
        - jnp.log(d.Z)
    )


def wright_mean(d: Wright, n_nodes=None):
    """Expected value E[p] of the Wright distribution."""
    n = n_nodes if n_nodes is not None else d.n_nodes
    A, B = 2 * d.Na, 2 * d.Nb
    return yfun(d.Ns, A, B, d.h, n) / d.Z


def wright_epq(d: Wright, n_nodes=None):
    """Expected value E[p·(1-p)] of the Wright distribution."""
    n = n_nodes if n_nodes is not None else d.n_nodes
    A, B = 2 * d.Na, 2 * d.Nb
    nodes, weights = gauss_jacobi(n, B - 1, A - 1)
    p_nodes = (nodes + 1) / 2
    values = p_nodes * (1 - p_nodes) * cfun(p_nodes, d.Ns, d.h)
    numerator = jnp.dot(weights, values) / 2 ** (A + B - 1)
    return numerator / d.Z


def wright_var(d: Wright, n_nodes=None):
    """Variance Var[p] = E[p]·E[q] - E[p·q] of the Wright distribution."""
    Ep = wright_mean(d, n_nodes)
    Epq = wright_epq(d, n_nodes)
    return Ep * (1 - Ep) - Epq


def wright_sfs(d: Wright, bins, n_nodes=None):
    """Site frequency spectrum: integrate PDF over each bin.

    Args:
        d: Wright distribution.
        bins: Array of bin edges on [0, 1] (length k+1 for k bins).
        n_nodes: Number of quadrature nodes per bin.

    Returns:
        (midpoints, integrals): bin midpoints and integrated density per bin.
    """
    n = n_nodes if n_nodes is not None else d.n_nodes
    bins = jnp.asarray(bins, dtype=jnp.float64)
    midpoints = (bins[:-1] + bins[1:]) / 2

    def _integrate_bin(lo, hi):
        # Gauss-Legendre on [lo, hi]: transform from [-1,1]
        gl_nodes, gl_weights = gauss_jacobi(n, 0.0, 0.0)
        # Map [-1,1] -> [lo, hi]
        t = (gl_nodes + 1) / 2 * (hi - lo) + lo
        scale = (hi - lo) / 2
        values = wright_pdf(d, t)
        return scale * jnp.dot(gl_weights, values)

    integrals = jax.vmap(_integrate_bin)(bins[:-1], bins[1:])
    return midpoints, integrals
