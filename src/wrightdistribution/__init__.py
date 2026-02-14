"""Wright distribution for population genetics, implemented in JAX."""

from .quadrature import gauss_jacobi
from .wright import (
    Wright,
    wright,
    wright_epq,
    wright_logpdf,
    wright_mean,
    wright_pdf,
    wright_sfs,
    wright_var,
)

__all__ = [
    "Wright",
    "gauss_jacobi",
    "wright",
    "wright_epq",
    "wright_logpdf",
    "wright_mean",
    "wright_pdf",
    "wright_sfs",
    "wright_var",
]
