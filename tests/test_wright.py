"""Unit tests for the Wright distribution, ported from Julia runtests.jl."""

import jax
import jax.numpy as jnp
import pytest

from wrightdistribution import (
    wright,
    wright_epq,
    wright_logpdf,
    wright_mean,
    wright_pdf,
    wright_sfs,
    wright_var,
)
from wrightdistribution.quadrature import gauss_jacobi


class TestNeutral:
    """Neutral case (s=0): distribution is Beta(2Nα, 2Nβ)."""

    def setup_method(self):
        Ne = 100
        s = 0.0
        h = 0.5
        u10 = 0.01  # q -> p
        u01 = 0.005  # p -> q
        self.u10 = u10
        self.u01 = u01
        self.Ne = Ne
        # Nα = 2*Ne*u10 = 2, Nβ = 2*Ne*u01 = 1
        # Distribution is Beta(A=4, B=2)
        self.d = wright(Ne * s, 2 * Ne * u10, 2 * Ne * u01, h)

    def test_mean(self):
        """Mean = u10/(u01+u10) for neutral case."""
        expected = self.u10 / (self.u01 + self.u10)
        assert wright_mean(self.d) == pytest.approx(expected, rel=1e-10)

    def test_variance(self):
        """Variance matches Beta(A, B) formula: A*B/((A+B)^2*(A+B+1))."""
        A = 2 * self.d.Na
        B = 2 * self.d.Nb
        expected = (A * B) / ((A + B) ** 2 * (A + B + 1))
        assert wright_var(self.d) == pytest.approx(expected, rel=1e-8)

    def test_epq(self):
        """E[p*q] matches Beta formula: A*B/((A+B)*(A+B+1))*(1/(A+B))."""
        A = 2 * self.d.Na
        B = 2 * self.d.Nb
        # For Beta(A,B): E[p(1-p)] = A*B / ((A+B)^2 * (A+B+1))  + ...
        # Actually E[p(1-p)] = E[p] - E[p^2] = A/(A+B) - A*(A+1)/((A+B)*(A+B+1))
        expected = A * B / ((A + B) * (A + B + 1))
        assert wright_epq(self.d) == pytest.approx(expected, rel=1e-8)


class TestSelection:
    """Selection case: mutation-selection balance."""

    def test_mutation_selection_balance(self):
        """q ≈ u/s for strong selection."""
        Ne = 1000
        s = 0.06
        u = 0.01
        h = 0.5
        d = wright(-Ne * (2 * s), Ne * u / 1000, Ne * u, h)
        q = 1 - wright_mean(d)
        assert u / s < q < u / s + 0.01


class TestPdf:
    """PDF properties."""

    @pytest.mark.parametrize(
        "Ns,Na,Nb,h",
        [
            (0.0, 1.0, 1.0, 0.5),
            (5.0, 0.5, 0.5, 0.5),
            (-10.0, 2.0, 1.0, 0.3),
            (50.0, 1.0, 1.0, 0.5),
        ],
    )
    def test_pdf_integrates_to_one(self, Ns, Na, Nb, h):
        """PDF integrates to 1 over [0,1] via independent Gauss-Legendre."""
        d = wright(Ns, Na, Nb, h)
        # Use Gauss-Legendre (alpha=beta=0) for independent verification
        n = 128
        nodes, weights = gauss_jacobi(n, 0.0, 0.0)
        p_nodes = (nodes + 1) / 2
        values = wright_pdf(d, p_nodes)
        integral = jnp.dot(weights, values) / 2
        assert float(integral) == pytest.approx(1.0, abs=1e-6)

    @pytest.mark.parametrize(
        "Ns,Na,Nb,h",
        [
            (0.0, 1.0, 1.0, 0.5),
            (5.0, 0.5, 0.5, 0.5),
            (-10.0, 2.0, 1.0, 0.3),
        ],
    )
    def test_logpdf_consistent(self, Ns, Na, Nb, h):
        """logpdf == log(pdf) at several points."""
        d = wright(Ns, Na, Nb, h)
        ps = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        for p in ps:
            log_val = wright_logpdf(d, p)
            pdf_val = wright_pdf(d, p)
            assert float(log_val) == pytest.approx(float(jnp.log(pdf_val)), abs=1e-10)

    def test_pdf_nonnegative(self):
        """PDF is non-negative everywhere."""
        d = wright(10.0, 0.5, 0.5, 0.5)
        ps = jnp.linspace(0.01, 0.99, 100)
        vals = jax.vmap(lambda p: wright_pdf(d, p))(ps)
        assert jnp.all(vals >= 0)


class TestSfs:
    """Site frequency spectrum tests."""

    def test_sfs_sums_to_one(self):
        """SFS bins sum to approximately 1."""
        d = wright(5.0, 1.0, 1.0, 0.5)
        bins = jnp.linspace(0.0, 1.0, 21)
        _, integrals = wright_sfs(d, bins)
        assert float(jnp.sum(integrals)) == pytest.approx(1.0, abs=1e-4)


class TestJaxCompat:
    """JAX compatibility: jit, vmap."""

    def test_jit_pdf(self):
        """wright_pdf works under jax.jit."""
        d = wright(5.0, 1.0, 1.0, 0.5)
        f = jax.jit(lambda p: wright_pdf(d, p))
        val = f(0.5)
        ref = wright_pdf(d, 0.5)
        assert float(val) == pytest.approx(float(ref), rel=1e-12)

    def test_jit_mean(self):
        """wright_mean works under jax.jit."""
        d = wright(5.0, 1.0, 1.0, 0.5)
        f = jax.jit(lambda: wright_mean(d))
        val = f()
        ref = wright_mean(d)
        assert float(val) == pytest.approx(float(ref), rel=1e-12)

    def test_vmap_pdf(self):
        """Batched PDF evaluation via jax.vmap."""
        d = wright(5.0, 1.0, 1.0, 0.5)
        ps = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
        batched = jax.vmap(lambda p: wright_pdf(d, p))(ps)
        individual = jnp.array([wright_pdf(d, p) for p in ps])
        assert jnp.allclose(batched, individual, rtol=1e-12)

    def test_vmap_over_params(self):
        """vmap over different parameter values."""
        Ns_vals = jnp.array([0.0, 1.0, 5.0, 10.0])

        def compute_mean(Ns):
            d = wright(Ns, 1.0, 1.0, 0.5)
            return wright_mean(d)

        means = jax.vmap(compute_mean)(Ns_vals)
        assert means.shape == (4,)
        # With s=0, mean should be 0.5 (symmetric)
        assert float(means[0]) == pytest.approx(0.5, rel=1e-8)
