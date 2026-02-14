"""AD gradient tests for the Wright distribution."""

import jax
import jax.numpy as jnp
import pytest

from wrightdistribution import wright, wright_logpdf, wright_mean, wright_var
from wrightdistribution.wright import zfun


def _finite_diff(f, x, eps=1e-5):
    """Central finite difference approximation of df/dx."""
    return (f(x + eps) - f(x - eps)) / (2 * eps)


def _assert_grad_close(ad, fd, *, rel=1e-4, abs_tol=1e-8):
    """Assert gradient values match, using abs tolerance for near-zero values."""
    assert ad == pytest.approx(fd, rel=rel, abs=abs_tol)


class TestLogpdfGradients:
    """Gradients of logpdf w.r.t. various parameters."""

    def test_grad_logpdf_wrt_p(self):
        """d(logpdf)/dp matches finite differences."""
        d = wright(5.0, 1.0, 1.0, 0.5)

        def f(p):
            return wright_logpdf(d, p)

        grad_f = jax.grad(f)
        for p in [0.2, 0.5, 0.8]:
            ad = float(grad_f(jnp.float64(p)))
            fd = float(_finite_diff(f, jnp.float64(p)))
            _assert_grad_close(ad, fd)

    def test_grad_logpdf_wrt_Ns(self):
        """d(logpdf)/dNs matches finite differences."""
        p = 0.5

        def f(Ns):
            d = wright(Ns, 1.0, 1.0, 0.5)
            return wright_logpdf(d, p)

        grad_f = jax.grad(f)
        for Ns in [0.0, 5.0, -10.0]:
            ad = float(grad_f(jnp.float64(Ns)))
            fd = float(_finite_diff(f, jnp.float64(Ns)))
            _assert_grad_close(ad, fd), (f"Ns={Ns}: ad={ad}, fd={fd}")

    def test_grad_logpdf_wrt_Na(self):
        """d(logpdf)/dNa matches finite differences."""
        p = 0.5

        def f(Na):
            d = wright(5.0, Na, 1.0, 0.5)
            return wright_logpdf(d, p)

        grad_f = jax.grad(f)
        for Na in [0.5, 1.0, 5.0]:
            ad = float(grad_f(jnp.float64(Na)))
            fd = float(_finite_diff(f, jnp.float64(Na)))
            _assert_grad_close(ad, fd), (f"Na={Na}: ad={ad}, fd={fd}")

    def test_grad_logpdf_wrt_h(self):
        """d(logpdf)/dh matches finite differences."""
        p = 0.5

        def f(h):
            d = wright(5.0, 1.0, 1.0, h)
            return wright_logpdf(d, p)

        grad_f = jax.grad(f)
        for h in [0.0, 0.25, 0.5, 0.75, 1.0]:
            ad = float(grad_f(jnp.float64(h)))
            fd = float(_finite_diff(f, jnp.float64(h)))
            _assert_grad_close(ad, fd), (f"h={h}: ad={ad}, fd={fd}")


class TestMomentGradients:
    """Gradients of mean and variance w.r.t. parameters."""

    def test_grad_mean_wrt_Ns(self):
        """d(mean)/dNs matches finite differences."""

        def f(Ns):
            d = wright(Ns, 1.0, 1.0, 0.5)
            return wright_mean(d)

        grad_f = jax.grad(f)
        for Ns in [0.0, 5.0, 20.0]:
            ad = float(grad_f(jnp.float64(Ns)))
            fd = float(_finite_diff(f, jnp.float64(Ns)))
            _assert_grad_close(ad, fd), (f"Ns={Ns}: ad={ad}, fd={fd}")

    def test_grad_mean_wrt_Na(self):
        """d(mean)/dNa matches finite differences."""

        def f(Na):
            d = wright(5.0, Na, 1.0, 0.5)
            return wright_mean(d)

        grad_f = jax.grad(f)
        for Na in [0.5, 1.0, 5.0]:
            ad = float(grad_f(jnp.float64(Na)))
            fd = float(_finite_diff(f, jnp.float64(Na)))
            _assert_grad_close(ad, fd), (f"Na={Na}: ad={ad}, fd={fd}")

    def test_grad_var_wrt_Ns(self):
        """d(var)/dNs matches finite differences."""

        def f(Ns):
            d = wright(Ns, 1.0, 1.0, 0.5)
            return wright_var(d)

        grad_f = jax.grad(f)
        for Ns in [0.0, 5.0, 20.0]:
            ad = float(grad_f(jnp.float64(Ns)))
            fd = float(_finite_diff(f, jnp.float64(Ns)))
            _assert_grad_close(ad, fd), (f"Ns={Ns}: ad={ad}, fd={fd}")

    def test_grad_var_wrt_Na(self):
        """d(var)/dNa matches finite differences."""

        def f(Na):
            d = wright(5.0, Na, 1.0, 0.5)
            return wright_var(d)

        grad_f = jax.grad(f)
        for Na in [0.5, 1.0, 5.0]:
            ad = float(grad_f(jnp.float64(Na)))
            fd = float(_finite_diff(f, jnp.float64(Na)))
            _assert_grad_close(ad, fd), (f"Na={Na}: ad={ad}, fd={fd}")


class TestZGradients:
    """Gradients of the normalization constant."""

    def test_grad_Z_wrt_Ns(self):
        """d(Z)/dNs matches finite differences."""

        def f(Ns):
            return zfun(Ns, 2.0, 2.0, 0.5)

        grad_f = jax.grad(f)
        for Ns in [0.0, 5.0, 20.0]:
            ad = float(grad_f(jnp.float64(Ns)))
            fd = float(_finite_diff(f, jnp.float64(Ns)))
            _assert_grad_close(ad, fd), (f"Ns={Ns}: ad={ad}, fd={fd}")

    def test_grad_Z_wrt_h(self):
        """d(Z)/dh matches finite differences."""

        def f(h):
            return zfun(10.0, 2.0, 2.0, h)

        grad_f = jax.grad(f)
        for h in [0.0, 0.5, 1.0]:
            ad = float(grad_f(jnp.float64(h)))
            fd = float(_finite_diff(f, jnp.float64(h)))
            _assert_grad_close(ad, fd), f"h={h}: ad={ad}, fd={fd}"


class TestHigherOrder:
    """Second-order derivatives."""

    def test_hessian_logpdf(self):
        """Hessian of logpdf w.r.t. (Ns, Na) is computable and symmetric."""

        def f(params):
            Ns, Na = params
            d = wright(Ns, Na, 1.0, 0.5)
            return wright_logpdf(d, 0.5)

        H = jax.hessian(f)(jnp.array([5.0, 1.0]))
        assert H.shape == (2, 2)
        # Hessian should be symmetric
        assert float(H[0, 1]) == pytest.approx(float(H[1, 0]), rel=1e-6)
