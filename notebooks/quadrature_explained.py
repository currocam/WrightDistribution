import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import pathlib
    import subprocess
    import time

    import marimo as mo

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import numpy as np
    import matplotlib.pyplot as plt

    import scienceplots
    plt.style.use("science")

    from wrightdistribution import (
        cfun,
        gauss_jacobi,
        wright,
        wright_epq,
        wright_mean,
        wright_pdf,
        wright_var,
    )

    return (
        cfun,
        gauss_jacobi,
        jnp,
        json,
        mo,
        np,
        pathlib,
        plt,
        subprocess,
        time,
        wright,
        wright_epq,
        wright_mean,
        wright_pdf,
        wright_var,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Why Gaussian Quadrature?

    ## The hidden integral

    The Wright distribution's unnormalized PDF is:

    $$
    f(p) \propto p^{2N\alpha - 1} \cdot (1-p)^{2N\beta - 1}
    \cdot \exp\!\bigl(-Ns \cdot p \cdot (2h + (1-2h)(2-p))\bigr)
    $$

    The **proportionality sign** $\propto$ hides a normalization integral that has
    **no closed form**:

    $$
    Z = \int_0^1 p^{2N\alpha - 1} \cdot (1-p)^{2N\beta - 1}
    \cdot \exp\!\bigl(-Ns \cdot p \cdot (2h + (1-2h)(2-p))\bigr)\, dp
    $$

    To turn $f(p)$ into a proper probability density, we need $\text{pdf}(p) = f(p) / Z$.
    Every evaluation of the PDF, mean, variance, or site frequency spectrum requires
    computing integrals like $Z$ numerically.

    This notebook explains step by step **why we chose Gauss-Jacobi quadrature** via
    the Golub-Welsch algorithm, and benchmarks our JAX implementation against Julia.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 1: The problem with singular integrands

    The integrand $p^{2N\alpha - 1} \cdot (1-p)^{2N\beta - 1} \cdot C(p)$ **blows up**
    at $p=0$ and $p=1$ whenever $2N\alpha < 1$ or $2N\beta < 1$
    (i.e., when mutation rates are small).

    Standard methods (trapezoidal rule, Simpson's rule) place nodes at or near the
    endpoints and fail catastrophically. Let's see this with a toy example.

    ### Toy integral: $\int_0^1 x^{-0.8} \cos(x)\, dx$

    This integral has a singularity at $x = 0$ (the integrand $\to \infty$),
    similar to what happens in the Wright distribution when $N\alpha$ is small.
    """)
    return


@app.cell
def _(gauss_jacobi, jnp):
    # Compute "exact" reference via high-order Gauss-Jacobi (n=200)
    # Integral: int_0^1 x^(-0.8) * cos(x) dx
    # Map [0,1] -> [-1,1]: x = (t+1)/2, singularity at x=0 -> t=-1
    # Weight: (1+t)^(-0.8) => Jacobi with alpha=0, beta=-0.8
    _nodes, _weights = gauss_jacobi(200, 0.0, -0.8)
    _x = (_nodes + 1) / 2
    _g = jnp.cos(_x)
    _scale = (0.5) ** 0.2  # from the change of variables
    toy_exact = float(_scale * jnp.dot(_weights, _g))
    return (toy_exact,)


@app.cell
def _(np, toy_exact):
    # Composite trapezoidal rule: struggles with the singularity
    _ns = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    trap_results = []
    for _n in _ns:
        _x = np.linspace(1e-15, 1.0, _n + 1)  # avoid x=0 exactly
        _f = _x ** (-0.8) * np.cos(_x)
        _h = 1.0 / _n
        _approx = _h * (0.5 * _f[0] + np.sum(_f[1:-1]) + 0.5 * _f[-1])
        _err = abs(_approx - toy_exact) / abs(toy_exact)
        trap_results.append((_n, _err))
    return (trap_results,)


@app.cell
def _(gauss_jacobi, jnp, toy_exact):
    # Gauss-Legendre (alpha=beta=0): no special handling of singularity
    _ns = [4, 8, 16, 32, 64, 128]
    gl_results = []
    for _n in _ns:
        _nodes, _weights = gauss_jacobi(_n, 0.0, 0.0)
        _x = (_nodes + 1) / 2
        _f = _x ** (-0.8) * jnp.cos(_x)
        _approx = float(jnp.dot(_weights, _f) / 2)
        _err = abs(_approx - toy_exact) / abs(toy_exact)
        gl_results.append((_n, _err))
    return (gl_results,)


@app.cell
def _(gauss_jacobi, jnp, toy_exact):
    # Gauss-Jacobi (alpha=0, beta=-0.8): singularity absorbed into weight
    _ns = [4, 8, 16, 32, 64, 128]
    gj_results = []
    for _n in _ns:
        _nodes, _weights = gauss_jacobi(_n, 0.0, -0.8)
        _x = (_nodes + 1) / 2
        _g = jnp.cos(_x)  # only the smooth part!
        _scale = (0.5) ** 0.2
        _approx = float(_scale * jnp.dot(_weights, _g))
        _err = abs(_approx - toy_exact) / abs(toy_exact)
        gj_results.append((_n, _err))
    return (gj_results,)


@app.cell
def _(gj_results, gl_results, plt, trap_results):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    _ax.loglog(
        [r[0] for r in trap_results],
        [r[1] for r in trap_results],
        "o-",
        label="Trapezoidal rule",
        color="C0",
    )
    _ax.loglog(
        [r[0] for r in gl_results],
        [r[1] for r in gl_results],
        "s-",
        label="Gauss-Legendre",
        color="C1",
    )
    _ax.loglog(
        [r[0] for r in gj_results],
        [max(r[1], 1e-16) for r in gj_results],
        "D-",
        label=r"Gauss-Jacobi ($\beta=-0.8$)",
        color="C2",
    )
    _ax.set_xlabel("Number of nodes")
    _ax.set_ylabel("Relative error")
    _ax.set_title(r"Convergence comparison: $\int_0^1 x^{-0.8}\cos(x)\,dx$")
    _ax.legend()
    _ax.grid(True, alpha=0.3)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### What we observe

    1. **Trapezoidal rule**: converges slowly ($\sim O(h)$).
       The singularity at $x=0$ prevents the usual $O(h^2)$ rate.

    2. **Gauss-Legendre** ($\alpha = \beta = 0$): much better than trapezoidal
       (exponential convergence for smooth integrands), but the singularity
       $x^{-0.8}$ degrades convergence to algebraic.

    3. **Gauss-Jacobi** ($\beta = -0.8$): the singularity is **absorbed into the
       weight function** $(1+t)^{-0.8}$, and the remaining integrand $\cos(\cdot)$
       is entire. Convergence is **spectral** -- with just 4-8 nodes we reach
       machine precision.

    This is exactly the situation with the Wright distribution: the boundary
    factors $p^{2N\alpha-1}(1-p)^{2N\beta-1}$ blow up at endpoints when mutation
    rates are small. Gauss-Jacobi absorbs them into the weight function.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 2: How Gaussian Quadrature Works

    **Gaussian quadrature** approximates weighted integrals:

    $$
    \int_a^b w(x)\, f(x)\, dx \approx \sum_{i=1}^{n} w_i\, f(x_i)
    $$

    The key insight: by choosing nodes $\{x_i\}$ **optimally** (not uniformly
    spaced), an $n$-point rule is **exact for all polynomials of degree
    $\leq 2n-1$** -- twice as many degrees of freedom as you'd expect.

    The optimal nodes turn out to be the **roots of the degree-$n$ orthogonal
    polynomial** for the weight function $w(x)$.

    | Weight $w(x)$ | Interval | Orthogonal polynomials | Quadrature name |
    |---|---|---|---|
    | $1$ | $[-1,1]$ | Legendre $P_n(x)$ | Gauss-Legendre |
    | $(1-x)^\alpha(1+x)^\beta$ | $[-1,1]$ | Jacobi $P_n^{(\alpha,\beta)}(x)$ | **Gauss-Jacobi** |
    | $e^{-x}$ | $[0,\infty)$ | Laguerre $L_n(x)$ | Gauss-Laguerre |
    | $e^{-x^2}$ | $(-\infty,\infty)$ | Hermite $H_n(x)$ | Gauss-Hermite |

    Gauss-Legendre is the special case $\alpha = \beta = 0$ (uniform weight).

    For the Wright distribution, the integrand weight $p^{A-1}(1-p)^{B-1}$ maps
    directly to Gauss-Jacobi with $\alpha = B-1$, $\beta = A-1$ after
    the change of variables $p = (t+1)/2$.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 3: The Golub-Welsch Algorithm

    Finding roots of orthogonal polynomials directly is numerically tricky.
    The **Golub-Welsch algorithm** (1969) converts this into a well-conditioned
    **eigenvalue problem**.

    ### The idea

    All orthogonal polynomials satisfy a **three-term recurrence**:

    $$\hat{P}_{k+1}(x) = (x - a_k)\,\hat{P}_k(x) - b_k\,\hat{P}_{k-1}(x)$$

    The recurrence coefficients $a_k$ and $b_k$ are known analytically for each
    polynomial family. For Jacobi polynomials $P_n^{(\alpha,\beta)}$:

    $$a_n = \frac{\beta^2 - \alpha^2}{(2n+\alpha+\beta)(2n+\alpha+\beta+2)}$$

    $$b_n = \frac{4\,n\,(n+\alpha)(n+\beta)(n+\alpha+\beta)}
    {(2n+\alpha+\beta)^2\,((2n+\alpha+\beta)^2 - 1)}$$

    ### From recurrence to eigenvalues

    Build the **symmetric tridiagonal Jacobi matrix** $J_n$ from these coefficients:

    $$J_n = \begin{pmatrix}
    a_0 & \sqrt{b_1} \\
    \sqrt{b_1} & a_1 & \sqrt{b_2} \\
     & \ddots & \ddots & \ddots \\
     & & \sqrt{b_{n-1}} & a_{n-1}
    \end{pmatrix}$$

    Then:
    - **Nodes** = eigenvalues of $J_n$
    - **Weights** = $\mu_0 \cdot v_{i,0}^2$ where $v_{i,0}$ is the first component
      of the $i$-th eigenvector and $\mu_0 = \int w(x)\,dx = 2^{\alpha+\beta+1}\,
      B(\alpha+1, \beta+1)$

    This is exactly what `jnp.linalg.eigh` computes -- and it's **automatically
    differentiable** via JAX.

    ### Step-by-step example

    Let's walk through the algorithm for $n=5$ with $\alpha=1.0$, $\beta=0.5$:
    """)
    return


@app.cell
def _(jnp):
    # Golub-Welsch step by step for n=5, alpha=1.0, beta=0.5
    _n = 5
    _alpha = jnp.float64(1.0)
    _beta = jnp.float64(0.5)
    _ab = _alpha + _beta

    # Step 1: Recurrence coefficients
    _i = jnp.arange(_n, dtype=jnp.float64)
    _denom = (2 * _i + _ab) * (2 * _i + _ab + 2)
    gw_diag = jnp.where(_denom == 0, 0.0, (_beta**2 - _alpha**2) / _denom)

    _j = jnp.arange(1, _n, dtype=jnp.float64)
    _numer = 4 * _j * (_j + _alpha) * (_j + _beta) * (_j + _ab)
    _s = 2 * _j + _ab
    _denom_off = _s**2 * (_s**2 - 1)
    _b = _numer / _denom_off
    gw_off_diag = jnp.sqrt(_b)

    # Step 2: Build the Jacobi matrix
    gw_J = (
        jnp.diag(gw_diag)
        + jnp.diag(gw_off_diag, 1)
        + jnp.diag(gw_off_diag, -1)
    )

    # Step 3: Eigendecomposition -> nodes and weights
    gw_nodes, _vecs = jnp.linalg.eigh(gw_J)

    from jax.scipy.special import betaln

    _log_mu0 = (_ab + 1) * jnp.log(2.0) + betaln(_alpha + 1, _beta + 1)
    _mu0 = jnp.exp(_log_mu0)
    gw_weights = _mu0 * _vecs[0, :] ** 2
    return gw_J, gw_diag, gw_nodes, gw_off_diag, gw_weights


@app.cell
def _(gauss_jacobi, gw_J, gw_diag, gw_nodes, gw_off_diag, gw_weights, jnp, mo):
    # Verify against gauss_jacobi
    _ref_nodes, _ref_weights = gauss_jacobi(5, 1.0, 0.5)
    _node_err = float(jnp.max(jnp.abs(gw_nodes - _ref_nodes)))
    _wt_err = float(jnp.max(jnp.abs(gw_weights - _ref_weights)))

    mo.md(f"""
    ### Golub-Welsch walkthrough: $n=5$, $\\alpha=1.0$, $\\beta=0.5$

    **Step 1 -- Diagonal** (recurrence $a_i$): `{[f'{float(x):.6f}' for x in gw_diag]}`

    **Step 1 -- Off-diagonal** ($\\sqrt{{b_i}}$): `{[f'{float(x):.6f}' for x in gw_off_diag]}`

    **Step 2 -- Jacobi matrix** $J_5$:

    ```
    {gw_J}
    ```

    **Step 3 -- Nodes** (eigenvalues of $J_5$): `{[f'{float(x):.6f}' for x in gw_nodes]}`

    **Step 3 -- Weights** ($\\mu_0 \\cdot v_{{i,0}}^2$): `{[f'{float(x):.6f}' for x in gw_weights]}`

    Verification against `gauss_jacobi(5, 1.0, 0.5)`:
    max node error = {_node_err:.2e}, max weight error = {_wt_err:.2e}
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Part 4: Application to the Wright distribution

    The normalization constant is:

    $$Z = \int_0^1 p^{A-1}(1-p)^{B-1} \cdot C(p)\, dp$$

    where $A = 2N\alpha$, $B = 2N\beta$, and
    $C(p) = \exp(-Ns \cdot p \cdot (2h + (1-2h)(2-p)))$.

    After the change of variables $p = (t+1)/2$ mapping $[0,1] \to [-1,1]$:

    $$Z = \frac{1}{2^{A+B-1}} \int_{-1}^{1}
    (1-t)^{B-1}(1+t)^{A-1} \cdot C\!\left(\tfrac{t+1}{2}\right) dt$$

    This is a Gauss-Jacobi integral with $\alpha = B-1$, $\beta = A-1$.
    The singular boundary terms are absorbed into the Jacobi weight, and the
    remaining integrand $C(p)$ is entire (a composition of polynomials and exp),
    so convergence is spectral.

    This is exactly what `zfun` computes:

    ```python
    nodes, weights = gauss_jacobi(n_nodes, B - 1, A - 1)
    p_nodes = (nodes + 1) / 2
    values = cfun(p_nodes, Ns, h)
    return jnp.dot(weights, values) / 2 ** (A + B - 1)
    ```
    """)
    return


@app.cell
def _(cfun, gauss_jacobi, jnp, plt):
    # Show convergence of Z for several Wright parameter regimes
    _cases = [
        {"Ns": 0.0, "Na": 1.0, "Nb": 1.0, "h": 0.5, "label": "Ns=0, Na=Nb=1 (easy)"},
        {"Ns": 10.0, "Na": 0.1, "Nb": 0.1, "h": 0.5, "label": "Ns=10, Na=Nb=0.1"},
        {
            "Ns": 100.0,
            "Na": 0.01,
            "Nb": 0.01,
            "h": 0.5,
            "label": "Ns=100, Na=Nb=0.01 (hard)",
        },
    ]

    _node_counts = list(range(4, 129, 4))
    _fig, _ax = plt.subplots(figsize=(8, 5))

    for _case in _cases:
        _A, _B = 2 * _case["Na"], 2 * _case["Nb"]
        # Reference: n=256
        _rn, _rw = gauss_jacobi(256, _B - 1, _A - 1)
        _rp = (_rn + 1) / 2
        _Z_ref = float(
            jnp.dot(_rw, cfun(_rp, _case["Ns"], _case["h"])) / 2 ** (_A + _B - 1)
        )

        _errs = []
        for _n in _node_counts:
            _nodes, _weights = gauss_jacobi(_n, _B - 1, _A - 1)
            _p = (_nodes + 1) / 2
            _Z = float(
                jnp.dot(_weights, cfun(_p, _case["Ns"], _case["h"]))
                / 2 ** (_A + _B - 1)
            )
            _rel = abs(_Z - _Z_ref) / abs(_Z_ref) if _Z_ref != 0 else abs(_Z - _Z_ref)
            _errs.append(max(_rel, 1e-16))

        _ax.semilogy(_node_counts, _errs, "o-", label=_case["label"], markersize=3)

    _ax.set_xlabel("Number of quadrature nodes")
    _ax.set_ylabel("Relative error in Z")
    _ax.set_title("Convergence of Wright normalization constant")
    _ax.legend(fontsize=8)
    _ax.grid(True, alpha=0.3)
    _ax.axhline(1e-12, color="gray", ls=":", alpha=0.5)
    _ax.annotate(
        "default n=64",
        xy=(64, 1e-16),
        fontsize=8,
        color="gray",
        ha="center",
    )
    _ax.axvline(64, color="gray", ls=":", alpha=0.3)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Part 5: Julia vs JAX Benchmark

    The Julia implementation (`WrightDistribution.jl`) uses **adaptive quadrature**
    via `QuadGK.jl` with an integration-by-parts trick to handle the endpoint
    singularities. The JAX implementation uses **fixed-node Gauss-Jacobi**
    (64 nodes by default).

    We compare both **accuracy** and **speed** across 240 parameter combinations.

    The Julia reference values are stored in `tests/julia_reference.json`
    (see also `tests/test_integration.py` for the cross-validation test suite).
    """)
    return


@app.cell
def _(json, pathlib):
    _path = pathlib.Path(__file__).parent.parent / "tests" / "julia_reference.json"
    with open(_path) as _f:
        reference_data = json.load(_f)
    return (reference_data,)


@app.cell
def _(reference_data, wright, wright_epq, wright_mean, wright_pdf, wright_var):
    jax_results = []
    for _case in reference_data:
        _d = wright(_case["Ns"], _case["Na"], _case["Nb"], _case["h"])
        jax_results.append(
            {
                "Z": float(_d.Z),
                "mean": float(wright_mean(_d)),
                "var": float(wright_var(_d)),
                "epq": float(wright_epq(_d)),
                "pdf_0.3": float(wright_pdf(_d, 0.3)),
                "pdf_0.7": float(wright_pdf(_d, 0.7)),
            }
        )
    return (jax_results,)


@app.cell
def _(jax_results, np, plt, reference_data):
    _quantities = ["Z", "mean", "var", "epq", "pdf_0.3", "pdf_0.7"]
    _fig, _axes = plt.subplots(2, 3, figsize=(14, 8))

    for _idx, _qty in enumerate(_quantities):
        _ax = _axes[_idx // 3, _idx % 3]
        _julia = np.array([c[_qty] for c in reference_data])
        _jax = np.array([c[_qty] for c in jax_results])

        _mask = np.abs(_julia) > 1e-15
        _rel_err = np.abs(_jax[_mask] - _julia[_mask]) / np.abs(_julia[_mask])

        _ax.hist(np.log10(_rel_err + 1e-20), bins=30, edgecolor="black", alpha=0.7)
        _ax.set_xlabel("log10(relative error)")
        _ax.set_ylabel("Count")
        _ax.set_title(f"{_qty} (n={int(np.sum(_mask))})")
        _ax.axvline(np.log10(1e-6), color="red", ls="--", alpha=0.5, label="1e-6")
        _ax.legend(fontsize=7)

    plt.suptitle(
        "JAX vs Julia: relative error distribution (240 parameter combos)", y=1.02
    )
    plt.tight_layout()
    _fig
    return


@app.cell
def _(jax_results, mo, np, reference_data):
    _quantities = ["Z", "mean", "var", "epq", "pdf_0.3", "pdf_0.7"]
    _rows = []
    for _qty in _quantities:
        _julia = np.array([c[_qty] for c in reference_data])
        _jax = np.array([c[_qty] for c in jax_results])
        _mask = np.abs(_julia) > 1e-15
        _rel_err = np.abs(_jax[_mask] - _julia[_mask]) / np.abs(_julia[_mask])
        _rows.append(
            f"| {_qty} | {np.max(_rel_err):.2e} | {np.median(_rel_err):.2e} "
            f"| {np.mean(_rel_err):.2e} |"
        )

    mo.md(
        f"""
    ### Accuracy summary (240 parameter combinations, 64 Gauss-Jacobi nodes)

    | Quantity | Max rel error | Median rel error | Mean rel error |
    |----------|--------------|-----------------|----------------|
    {chr(10).join(_rows)}
    """
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Speed comparison

    We time both implementations on the same parameter grid.

    - **JAX**: timed after JIT warmup (subsequent calls use cached compilation)
    - **Julia**: timed via `subprocess` after a warmup call
    """)
    return


@app.cell
def _(time, wright, wright_mean, wright_var):
    # JAX timing (after JIT warmup)
    _d = wright(1.0, 1.0, 1.0, 0.5)
    _ = wright_mean(_d)
    _ = wright_var(_d)

    _Ns_vals = [0.0, 0.1, 1.0, 10.0, 100.0]
    _Na_vals = [0.01, 0.1, 1.0, 10.0]
    _h_vals = [0.0, 0.5, 1.0]

    _start = time.perf_counter()
    _count = 0
    for _Ns in _Ns_vals:
        for _Na in _Na_vals:
            for _Nb in _Na_vals:
                for _h in _h_vals:
                    _d = wright(_Ns, _Na, _Nb, _h)
                    _ = wright_mean(_d)
                    _ = wright_var(_d)
                    _count += 1
    _elapsed = time.perf_counter() - _start

    jax_timing = {
        "total_s": _elapsed,
        "n_cases": _count,
        "per_case_ms": _elapsed / _count * 1000,
    }
    return (jax_timing,)


@app.cell
def _(json, pathlib, subprocess):
    _julia_code = r"""
    using WrightDistribution
    Ns_vals = [0.0, 0.1, 1.0, 10.0, 100.0]
    Na_vals = [0.01, 0.1, 1.0, 10.0]
    h_vals  = [0.0, 0.5, 1.0]
    # Warmup
    d = Wright(1.0, 1.0, 1.0, 0.5)
    mean(d); var(d)
    # Timed run
    t0 = time_ns()
    count = 0
    for Ns in Ns_vals, Na in Na_vals, Nb in Na_vals, h in h_vals
        d = Wright(Ns, Na, Nb, h)
        mean(d); var(d)
        count += 1
    end
    elapsed_ms = (time_ns() - t0) / 1e6
    println("{\"total_ms\": $elapsed_ms, \"n_cases\": $count, \"per_case_ms\": $(elapsed_ms/count)}")
    """
    _julia_project = str(
        pathlib.Path(__file__).parent.parent / "WrightDistribution"
    )
    try:
        _result = subprocess.run(
            ["julia", f"--project={_julia_project}", "-e", _julia_code],
            capture_output=True,
            text=True,
            timeout=300,
        )
        _output = _result.stdout.strip()
        julia_timing = json.loads(_output)
    except Exception as _e:
        julia_timing = {
            "error": str(_e),
            "total_ms": None,
            "n_cases": None,
            "per_case_ms": None,
        }
    return (julia_timing,)


@app.cell
def _(jax_timing, julia_timing, mo):
    _jax_per = jax_timing["per_case_ms"]
    _julia_per = julia_timing.get("per_case_ms")

    if _julia_per:
        _julia_str = f"{_julia_per:.2f} ms"
        _julia_total = f"{julia_timing['total_ms'] / 1000:.2f} s"
        _ratio = f"{_julia_per / _jax_per:.1f}x"
    else:
        _julia_str = f"Error: {julia_timing.get('error', 'unknown')}"
        _julia_total = "N/A"
        _ratio = "N/A"

    mo.md(f"""
    ### Timing results ({jax_timing['n_cases']} parameter combinations)

    | Implementation | Total time | Per case | Method |
    |---|---|---|---|
    | **JAX** (Gauss-Jacobi, 64 nodes) | {jax_timing['total_s']:.2f} s | {_jax_per:.2f} ms | Fixed quadrature, JIT-compiled |
    | **Julia** (QuadGK, adaptive) | {_julia_total} | {_julia_str} | Adaptive quadrature + IBP |

    **Ratio** (Julia / JAX per case): {_ratio}

    *JAX timing excludes initial JIT compilation. Julia timing excludes precompilation warmup.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Discussion

    ### Why Gauss-Jacobi beats adaptive quadrature here

    1. **Fixed cost, predictable performance**: 64 Gauss-Jacobi nodes give spectral
       convergence for the entire parameter range. No adaptive refinement needed
       because the singularity structure is known analytically.

    2. **Differentiability**: The fixed-node scheme is a simple dot product
       $Z = \mathbf{w}^\top \mathbf{f}$, trivially differentiable via JAX's AD.
       Adaptive quadrature (QuadGK) uses control flow (while loops, error estimates)
       that is harder to differentiate.

    3. **Vectorization**: The fixed grid allows `jax.vmap` over parameter batches.
       QuadGK processes one integral at a time.

    ### Trade-offs

    - Julia's adaptive approach can **guarantee** a specific error tolerance.
      The fixed-node approach relies on choosing $n$ large enough.
    - For very large $Ns$, the integrand $C(p)$ oscillates more, potentially
      needing more nodes. The default $n=64$ is conservative.
    - Julia's integration-by-parts trick analytically removes some difficulty,
      reducing the adaptive integrator's work.

    ### Summary

    The $\propto$ sign in the Wright PDF conceals an integral that must be computed
    numerically. Gauss-Jacobi quadrature is the natural choice because the
    integrand's boundary singularity matches the Jacobi weight function exactly.
    The Golub-Welsch algorithm computes nodes and weights via eigendecomposition,
    which slots naturally into JAX's autodiff framework -- making the entire
    distribution (normalization, moments, gradients) fully differentiable.
    """)
    return


if __name__ == "__main__":
    app.run()
