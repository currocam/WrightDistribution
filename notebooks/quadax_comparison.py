import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import time

    import marimo as mo

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import matplotlib.pyplot as plt
    import quadax

    from wrightdistribution import cfun, gauss_jacobi, wright, wright_mean

    return cfun, jax, jnp, mo, quadax, time, wright, wright_mean


@app.cell
def _(mo):
    mo.md(r"""
    # quadax vs Gauss-Jacobi for the Wright Distribution

    [quadax](https://github.com/f0uriest/quadax) is a JAX-native adaptive
    quadrature library -- the JAX equivalent of `scipy.integrate.quad`.
    It provides `quadgk` (Gauss-Kronrod) and `quadts` (tanh-sinh), both
    fully `jit`-able, `vmap`-able, and differentiable.

    **Question**: Can we use `quadax.quadgk` instead of our custom Gauss-Jacobi
    quadrature for the Wright distribution?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Test 1: Accuracy on the normalization integral

    The Wright normalization constant is:

    $$Z = \int_0^1 p^{A-1}(1-p)^{B-1} \cdot C(p)\, dp$$

    We compare three methods:
    - **Gauss-Jacobi** (64 nodes): absorbs the $p^{A-1}(1-p)^{B-1}$ singularity
      into the weight function
    - **quadax.quadgk**: adaptive Gauss-Kronrod, integrates the full integrand
    - **quadax.quadts**: adaptive tanh-sinh, designed for endpoint singularities
    """)
    return


@app.cell
def _(cfun, mo, quadax, wright):
    _cases = [
        {"Ns": 0.0, "Na": 1.0, "Nb": 1.0, "h": 0.5, "label": "Ns=0, Na=Nb=1 (easy)"},
        {
            "Ns": 10.0,
            "Na": 0.1,
            "Nb": 0.1,
            "h": 0.5,
            "label": "Ns=10, Na=Nb=0.1 (singular)",
        },
        {
            "Ns": 100.0,
            "Na": 0.01,
            "Nb": 0.01,
            "h": 0.5,
            "label": "Ns=100, Na=Nb=0.01 (very singular)",
        },
        {
            "Ns": 10.0,
            "Na": 1.0,
            "Nb": 1.0,
            "h": 0.0,
            "label": "Ns=10, Na=Nb=1, h=0",
        },
        {
            "Ns": -10.0,
            "Na": 2.0,
            "Nb": 1.0,
            "h": 0.3,
            "label": "Ns=-10, Na=2, Nb=1",
        },
    ]

    _rows = []
    for _c in _cases:
        _A, _B = 2 * _c["Na"], 2 * _c["Nb"]

        # Our Gauss-Jacobi
        _d = wright(_c["Ns"], _c["Na"], _c["Nb"], _c["h"])
        _Z_gj = float(_d.Z)

        # quadax.quadgk
        def _integrand(_p, _A=_A, _B=_B, _Ns=_c["Ns"], _h=_c["h"]):
            return _p ** (_A - 1) * (1 - _p) ** (_B - 1) * cfun(_p, _Ns, _h)

        _Z_qgk, _info_qgk = quadax.quadgk(_integrand, [1e-12, 1 - 1e-12])
        _Z_qgk = float(_Z_qgk)
        _neval_qgk = int(_info_qgk.neval)

        # quadax.quadts
        _Z_qts, _info_qts = quadax.quadts(_integrand, [1e-12, 1 - 1e-12])
        _Z_qts = float(_Z_qts)
        _neval_qts = int(_info_qts.neval)

        # Use Julia-validated Gauss-Jacobi as reference
        _rel_qgk = abs(_Z_qgk - _Z_gj) / abs(_Z_gj) if _Z_gj != 0 else 0
        _rel_qts = abs(_Z_qts - _Z_gj) / abs(_Z_gj) if _Z_gj != 0 else 0

        _rows.append(
            f"| {_c['label']} | {_Z_gj:.6e} | {_Z_qgk:.6e} ({_neval_qgk} evals, "
            f"{'OK' if _rel_qgk < 1e-4 else f'**{_rel_qgk:.1e}**'}) | "
            f"{_Z_qts:.6e} ({_neval_qts} evals, "
            f"{'OK' if _rel_qts < 1e-4 else f'**{_rel_qts:.1e}**'}) |"
        )

    accuracy_table = mo.md(
        f"""
    ### Results

    | Case | Gauss-Jacobi (64 nodes) | quadax.quadgk | quadax.quadts |
    |------|------------------------|---------------|---------------|
    {chr(10).join(_rows)}

    **Gauss-Jacobi values are validated against Julia's QuadGK across 240 parameter
    combinations** (see `tests/test_integration.py`).

    When $N\\alpha$ or $N\\beta$ is small (e.g., 0.1 or 0.01), the integrand has a
    strong endpoint singularity $p^{{-0.8}}$ or $p^{{-0.98}}$. `quadax.quadgk` and
    `quadax.quadts` integrate the **full singular integrand** directly and struggle,
    while Gauss-Jacobi absorbs the singularity into the weight function.
    """
    )
    accuracy_table
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Test 2: Differentiability

    Both approaches support `jax.grad`. Let's verify they agree on $dZ/dNs$.
    """)
    return


@app.cell
def _(cfun, jax, jnp, mo, quadax):
    from wrightdistribution.wright import zfun

    def _Z_via_quadgk(_Ns):
        _A, _B = 2.0, 2.0

        def _integrand(_p):
            return _p ** (_A - 1) * (1 - _p) ** (_B - 1) * cfun(_p, _Ns, 0.5)

        _y, _ = quadax.quadgk(_integrand, [1e-12, 1 - 1e-12])
        return _y

    def _Z_via_gj(_Ns):
        return zfun(_Ns, 2.0, 2.0, 0.5)

    _grad_quadgk = float(jax.grad(_Z_via_quadgk)(jnp.float64(5.0)))
    _grad_gj = float(jax.grad(_Z_via_gj)(jnp.float64(5.0)))

    mo.md(f"""
    ### $dZ/dNs$ at $Ns=5$, $A=B=2$, $h=0.5$ (no singularity)

    | Method | Gradient |
    |--------|----------|
    | Gauss-Jacobi | {_grad_gj:.10f} |
    | quadax.quadgk | {_grad_quadgk:.10f} |
    | Difference | {abs(_grad_quadgk - _grad_gj):.2e} |

    Both approaches are differentiable and agree well on this non-singular case.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Test 3: Speed

    We time both approaches on the full parameter grid (240 cases),
    computing `wright` + `wright_mean` for each.
    """)
    return


@app.cell
def _(cfun, mo, quadax, time, wright, wright_mean):
    # --- Gauss-Jacobi timing ---
    _d = wright(1.0, 1.0, 1.0, 0.5)
    _ = wright_mean(_d)

    _Ns_vals = [0.0, 0.1, 1.0, 10.0, 100.0]
    _Na_vals = [0.01, 0.1, 1.0, 10.0]
    _h_vals = [0.0, 0.5, 1.0]

    _t0 = time.perf_counter()
    _count = 0
    for _Ns in _Ns_vals:
        for _Na in _Na_vals:
            for _Nb in _Na_vals:
                for _h in _h_vals:
                    _d = wright(_Ns, _Na, _Nb, _h)
                    _ = wright_mean(_d)
                    _count += 1
    _gj_elapsed = time.perf_counter() - _t0

    # --- quadax timing (subset -- full grid is too slow) ---
    def _mean_quadgk(_Ns, _Na, _Nb, _h):
        _A, _B = 2 * _Na, 2 * _Nb

        def _unnorm(_p):
            return _p ** (_A - 1) * (1 - _p) ** (_B - 1) * cfun(_p, _Ns, _h)

        def _unnorm_p(_p):
            return _p * _p ** (_A - 1) * (1 - _p) ** (_B - 1) * cfun(_p, _Ns, _h)

        _Z, _ = quadax.quadgk(_unnorm, [1e-12, 1 - 1e-12])
        _Y, _ = quadax.quadgk(_unnorm_p, [1e-12, 1 - 1e-12])
        return _Y / _Z

    # Warmup
    _ = _mean_quadgk(1.0, 1.0, 1.0, 0.5)

    # Time a 20-case subset to estimate
    _t0 = time.perf_counter()
    _qcount = 0
    for _Ns in [0.0, 1.0, 10.0, 100.0]:
        for _Na in [0.1, 1.0]:
            for _Nb in [0.1, 1.0]:
                for _h in [0.5]:
                    _ = _mean_quadgk(_Ns, _Na, _Nb, _h)
                    _qcount += 1
    _qgk_elapsed = time.perf_counter() - _t0
    _qgk_per_case = _qgk_elapsed / _qcount * 1000
    _gj_per_case = _gj_elapsed / _count * 1000
    _ratio = _qgk_per_case / _gj_per_case

    mo.md(f"""
    ### Timing results

    | Method | Cases timed | Per case | Notes |
    |--------|-------------|----------|-------|
    | **Gauss-Jacobi** (64 nodes) | {_count} | **{_gj_per_case:.2f} ms** | Fixed-cost eigendecomposition |
    | **quadax.quadgk** (adaptive) | {_qcount} | **{_qgk_per_case:.1f} ms** | Adaptive, varies by difficulty |

    **Ratio**: quadax is ~{_ratio:.0f}x slower per case.

    *quadax timed on a {_qcount}-case subset (full 240 would take ~{240*_qgk_per_case/1000:.0f}s).*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Conclusion

    | Criterion | Gauss-Jacobi (ours) | quadax.quadgk |
    |-----------|-------------------|---------------|
    | **Singular integrands** | Handles exactly (absorbed into weight) | Struggles when $N\alpha, N\beta < 0.5$ |
    | **Speed** | ~2 ms/case | ~200-400 ms/case |
    | **Differentiability** | Full AD via JAX | Full AD via JAX |
    | **JIT / vmap** | Yes | Yes |
    | **Adaptive error control** | No (fixed 64 nodes) | Yes (but unreliable for singularities) |

    **Verdict**: `quadax` is a great general-purpose JAX quadrature library, but for
    the Wright distribution specifically, our Gauss-Jacobi approach wins on both
    **accuracy** (handles boundary singularities exactly) and **speed** (~100x faster).

    The key insight is that we *know* the singularity structure of our integrand
    analytically -- it's always $p^{A-1}(1-p)^{B-1}$ -- so we can factor it into
    the quadrature weight function. `quadax` treats the integrand as a black box
    and has to discover and adapt to the singularity numerically.
    """)
    return


if __name__ == "__main__":
    app.run()
