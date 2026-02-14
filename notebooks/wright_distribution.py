import marimo

__generated_with = "0.19.11"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    import jax

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    import matplotlib.pyplot as plt

    from wrightdistribution import (
        wright,
        wright_mean,
        wright_pdf,
        wright_sfs,
        wright_var,
    )

    return (
        jax,
        jnp,
        mo,
        plt,
        wright,
        wright_mean,
        wright_pdf,
        wright_sfs,
        wright_var,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # The Wright Distribution

    The **Wright distribution** describes the stationary distribution of allele
    frequencies in a finite population under the combined effects of mutation,
    selection, and genetic drift. Its unnormalized density on $[0, 1]$ is:

    $$
    f(p) \propto p^{2N\alpha - 1} \cdot (1-p)^{2N\beta - 1}
    \cdot \exp\!\bigl(-Ns \cdot p \cdot (2h + (1-2h)(2-p))\bigr)
    $$

    where:
    - $p$ is the allele frequency
    - $Ns$ is the scaled selection coefficient
    - $N\alpha$, $N\beta$ are the scaled mutation rates
    - $h$ is the dominance coefficient

    This implementation uses **JAX** with Gauss-Jacobi quadrature,
    making it fully differentiable via `jax.grad`.
    """)
    return


@app.cell
def _(mo):
    Ns_slider = mo.ui.slider(-50.0, 50.0, value=0.0, step=0.5, label="Ns")
    Na_slider = mo.ui.slider(0.1, 10.0, value=1.0, step=0.1, label="Na")
    Nb_slider = mo.ui.slider(0.1, 10.0, value=1.0, step=0.1, label="Nb")
    h_slider = mo.ui.slider(0.0, 1.0, value=0.5, step=0.05, label="h")
    mo.hstack([Ns_slider, Na_slider, Nb_slider, h_slider])
    return Na_slider, Nb_slider, Ns_slider, h_slider


@app.cell
def _(
    Na_slider,
    Nb_slider,
    Ns_slider,
    h_slider,
    jax,
    jnp,
    plt,
    wright,
    wright_mean,
    wright_pdf,
    wright_var,
):
    _d = wright(Ns_slider.value, Na_slider.value, Nb_slider.value, h_slider.value)
    _ps = jnp.linspace(0.001, 0.999, 500)
    _density = jax.vmap(lambda p: wright_pdf(_d, p))(_ps)

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(_ps, _density, "k-", lw=2)
    _ax.fill_between(_ps, _density, alpha=0.2)
    _m = float(wright_mean(_d))
    _ax.axvline(_m, color="red", ls="--", label=f"mean = {_m:.4f}")
    _v = float(wright_var(_d))
    _ax.set_xlabel("Allele frequency p")
    _ax.set_ylabel("Density")
    _ax.set_title(
        f"Wright(Ns={Ns_slider.value}, Na={Na_slider.value}, "
        f"Nb={Nb_slider.value}, h={h_slider.value})  "
        f"var={_v:.4f}"
    )
    _ax.legend()
    _ax.set_xlim(0, 1)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Special cases

    ### 1. Neutral evolution ($Ns = 0$)

    When there is no selection, the Wright distribution reduces to a
    **Beta distribution** $\mathrm{Beta}(2N\alpha, 2N\beta)$.
    The mean allele frequency is $\alpha / (\alpha + \beta)$.

    ### 2. Mutation-selection balance

    Under strong selection against an allele ($Ns \ll 0$),
    the equilibrium frequency satisfies $q \approx u/s$
    (the classical mutation-selection balance result).

    ### 3. Dominance

    The parameter $h$ controls dominance:
    - $h = 0.5$: additive (co-dominant) selection
    - $h = 0$: the selected allele is fully recessive
    - $h = 1$: the selected allele is fully dominant
    """)
    return


@app.cell
def _(jax, jnp, plt, wright, wright_pdf):
    _ps = jnp.linspace(0.001, 0.999, 500)
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4))

    # Neutral
    _ax = _axes[0]
    for _Na, _Nb in [(0.5, 0.5), (1.0, 1.0), (2.0, 1.0), (5.0, 2.0)]:
        _d = wright(0.0, _Na, _Nb, 0.5)
        _density = jax.vmap(lambda p, d_=_d: wright_pdf(d_, p))(_ps)
        _ax.plot(_ps, _density, label=f"Na={_Na}, Nb={_Nb}")
    _ax.set_title("Neutral (Ns=0)")
    _ax.set_xlabel("p")
    _ax.legend(fontsize=8)

    # Selection strength
    _ax = _axes[1]
    for _Ns in [0.0, 5.0, 20.0, -5.0, -20.0]:
        _d = wright(_Ns, 1.0, 1.0, 0.5)
        _density = jax.vmap(lambda p, d_=_d: wright_pdf(d_, p))(_ps)
        _ax.plot(_ps, _density, label=f"Ns={_Ns}")
    _ax.set_title("Selection strength (Na=Nb=1)")
    _ax.set_xlabel("p")
    _ax.legend(fontsize=8)

    # Dominance
    _ax = _axes[2]
    for _h in [0.0, 0.25, 0.5, 0.75, 1.0]:
        _d = wright(10.0, 1.0, 1.0, _h)
        _density = jax.vmap(lambda p, d_=_d: wright_pdf(d_, p))(_ps)
        _ax.plot(_ps, _density, label=f"h={_h}")
    _ax.set_title("Dominance (Ns=10)")
    _ax.set_xlabel("p")
    _ax.legend(fontsize=8)

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Site Frequency Spectrum

    The **site frequency spectrum** (SFS) is the distribution of allele
    frequencies across sites. For a Wright distribution, each bin of the SFS
    is the integral of the PDF over that frequency interval.
    """)
    return


@app.cell
def _(jnp, plt, wright, wright_sfs):
    _bins = jnp.linspace(0.0, 1.0, 21)
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4))

    # Different selection
    _ax = _axes[0]
    for _Ns in [0.0, 5.0, 20.0, -10.0]:
        _d = wright(_Ns, 1.0, 1.0, 0.5)
        _midpoints, _integrals = wright_sfs(_d, _bins, n_nodes=32)
        _ax.bar(
            _midpoints + _Ns * 0.002,
            _integrals,
            width=0.04,
            alpha=0.7,
            label=f"Ns={_Ns}",
        )
    _ax.set_xlabel("Allele frequency")
    _ax.set_ylabel("Proportion of sites")
    _ax.set_title("SFS under different selection")
    _ax.legend(fontsize=8)

    # Different mutation rates
    _ax = _axes[1]
    for _Na, _Nb in [(0.5, 0.5), (1.0, 1.0), (5.0, 1.0)]:
        _d = wright(0.0, _Na, _Nb, 0.5)
        _midpoints, _integrals = wright_sfs(_d, _bins, n_nodes=32)
        _ax.bar(
            _midpoints + _Na * 0.005,
            _integrals,
            width=0.04,
            alpha=0.7,
            label=f"Na={_Na}, Nb={_Nb}",
        )
    _ax.set_xlabel("Allele frequency")
    _ax.set_ylabel("Proportion of sites")
    _ax.set_title("SFS under different mutation rates")
    _ax.legend(fontsize=8)

    plt.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Automatic Differentiation

    Because the implementation uses JAX with differentiable Gauss-Jacobi
    quadrature (via the Golub-Welsch algorithm), we can compute gradients
    of **any** quantity with respect to **any** parameter.
    """)
    return


@app.cell
def _(jax, jnp, plt, wright, wright_mean):
    _Ns_range = jnp.linspace(-20.0, 20.0, 200)

    def _mean_of_Ns(_Ns):
        _d = wright(_Ns, 1.0, 1.0, 0.5)
        return wright_mean(_d)

    _grad_mean = jax.vmap(jax.grad(_mean_of_Ns))(_Ns_range)
    _means = jax.vmap(_mean_of_Ns)(_Ns_range)

    _fig, (_ax1, _ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    _ax1.plot(_Ns_range, _means, "k-", lw=2)
    _ax1.set_ylabel("E[p]")
    _ax1.set_title("Mean allele frequency and its gradient w.r.t. Ns")
    _ax1.axhline(0.5, color="gray", ls=":", alpha=0.5)

    _ax2.plot(_Ns_range, _grad_mean, "b-", lw=2)
    _ax2.set_xlabel("Ns")
    _ax2.set_ylabel("dE[p]/dNs")
    _ax2.axhline(0, color="gray", ls=":", alpha=0.5)

    plt.tight_layout()
    _fig
    return


if __name__ == "__main__":
    app.run()
