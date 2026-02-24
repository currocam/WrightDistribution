"""Integration tests comparing JAX results against Julia reference values."""

import json
import pathlib

import jax.numpy as jnp
import pytest

from wrightdistribution import (
    wright,
    wright_epq,
    wright_logpdf,
    wright_mean,
    wright_pdf,
    wright_var,
)

REFERENCE_FILE = pathlib.Path(__file__).parent / "julia_reference.json"


@pytest.fixture(scope="module")
def reference_data():
    with open(REFERENCE_FILE) as f:
        return json.load(f)


def _case_id(case):
    return f"Ns={case['Ns']}_Na={case['Na']}_Nb={case['Nb']}_h={case['h']}"


class TestJuliaReference:
    """Compare JAX implementation against Julia WrightDistribution.jl."""

    def test_reference_file_exists(self):
        assert REFERENCE_FILE.exists(), (
            f"Reference file not found: {REFERENCE_FILE}. "
            "Generate it by running the Julia script."
        )

    def test_Z(self, reference_data):
        """Normalization constant matches Julia."""
        for case in reference_data:
            d = wright(case["Ns"], case["Na"], case["Nb"], case["h"])
            assert float(d.Z) == pytest.approx(case["Z"], rel=1e-6), _case_id(case)

    def test_mean(self, reference_data):
        """Mean matches Julia."""
        for case in reference_data:
            d = wright(case["Ns"], case["Na"], case["Nb"], case["h"])
            assert float(wright_mean(d)) == pytest.approx(case["mean"], rel=1e-6), (
                _case_id(case)
            )

    def test_var(self, reference_data):
        """Variance matches Julia."""
        for case in reference_data:
            d = wright(case["Ns"], case["Na"], case["Nb"], case["h"])
            jax_var = float(wright_var(d))
            julia_var = case["var"]
            # Use absolute tolerance for near-zero variances
            if abs(julia_var) < 1e-10:
                assert jax_var == pytest.approx(julia_var, abs=1e-6), _case_id(case)
            else:
                assert jax_var == pytest.approx(julia_var, rel=1e-4), _case_id(case)

    def test_pdf(self, reference_data):
        """PDF at p=0.3 and p=0.7 matches Julia."""
        for case in reference_data:
            d = wright(case["Ns"], case["Na"], case["Nb"], case["h"])
            for p_str, p_val in [("pdf_0.3", 0.3), ("pdf_0.7", 0.7)]:
                jax_val = float(wright_pdf(d, p_val))
                julia_val = case[p_str]
                if julia_val == 0.0 or abs(julia_val) < 1e-300:
                    continue
                assert jax_val == pytest.approx(julia_val, rel=1e-4), (
                    f"{_case_id(case)} p={p_val}"
                )

    def test_logpdf(self, reference_data):
        """logpdf at p=0.3 and p=0.7 matches Julia."""
        for case in reference_data:
            d = wright(case["Ns"], case["Na"], case["Nb"], case["h"])
            for p_str, p_val in [("logpdf_0.3", 0.3), ("logpdf_0.7", 0.7)]:
                jax_val = float(wright_logpdf(d, p_val))
                julia_val = case[p_str]
                if not jnp.isfinite(julia_val):
                    continue
                assert jax_val == pytest.approx(julia_val, abs=1e-3), (
                    f"{_case_id(case)} p={p_val}"
                )

    def test_epq(self, reference_data):
        """E[pq] matches Julia."""
        for case in reference_data:
            d = wright(case["Ns"], case["Na"], case["Nb"], case["h"])
            jax_val = float(wright_epq(d))
            julia_val = case["epq"]
            if abs(julia_val) < 1e-10:
                assert jax_val == pytest.approx(julia_val, abs=1e-6), _case_id(case)
            else:
                assert jax_val == pytest.approx(julia_val, rel=1e-4), _case_id(case)
