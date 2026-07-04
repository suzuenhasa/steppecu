import marimo

__generated_with = "0.10.0"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Steppe — from genotypes to a DataFrame

        A short worked example: **build an f2 cache from raw genotypes**, then run **qpAdm** over
        it and see what the results are actually *for* — because every Steppe result comes back as
        a **pandas DataFrame** you can filter, sort, plot, and join.

        *This runs Steppe, so it needs a CUDA-13 GPU. It defaults to the bundled 10-population
        example; point `STEPPE_F2_DIR` at a bigger cache (and `STEPPE_GENO_PREFIX` at your own
        genotypes) for more to explore.*
        """
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import itertools
    import json
    import os
    import time

    import matplotlib
    matplotlib.use("Agg")  # headless-safe; marimo renders the figure object itself
    import matplotlib.pyplot as plt
    import pandas as pd

    import steppe
    return itertools, json, matplotlib, os, pd, plt, steppe, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 0. Where the cache comes from — `extract_f2`

        Steppe is *precompute once, fit many*. The one expensive step is a single streaming pass
        over the raw genotypes that builds the **f2 cache**; every fit afterward is cheap. Set
        `STEPPE_GENO_PREFIX` to a genotype triple (`.geno/.snp/.ind`) to run it:
        """
    )
    return


@app.cell
def _(json, os, steppe, time):
    PREFIX = os.environ.get("STEPPE_GENO_PREFIX", "/path/to/your/genotypes")
    BUILD_POPS = [
        "Czechia_EBA_CordedWare", "England_BellBeaker", "Russia_Samara_EBA_Yamnaya", "Turkey_N",
        "Mbuti", "Israel_Natufian", "Iran_GanjDareh_N", "Han", "Papuan", "Karitiana",
    ]
    BUILD_DIR = os.path.expanduser("~/nb_built_f2")

    build_meta, build_secs = None, None
    if os.path.exists(PREFIX + ".geno"):
        _t = time.perf_counter()
        steppe.extract_f2(PREFIX, pops=BUILD_POPS, out=BUILD_DIR, maxmiss=0.0, device=0)
        build_secs = time.perf_counter() - _t
        build_meta = json.load(open(f"{BUILD_DIR}/meta.json"))
    return BUILD_POPS, PREFIX, build_meta, build_secs


@app.cell(hide_code=True)
def _(BUILD_POPS, build_meta, build_secs, mo):
    if build_meta is None:
        _msg = ("*(Skipped — set `STEPPE_GENO_PREFIX` to a genotype prefix to build a cache from "
                "scratch. The rest of the notebook uses a prebuilt cache below.)*")
    else:
        _msg = (
            f"Built an f2 cache for **{len(BUILD_POPS)} populations in {build_secs:.1f} s** — kept "
            f"**{build_meta['n_snp_kept']:,} of {build_meta['n_snp_total']:,} SNPs** across "
            f"**{build_meta['n_block']} blocks**. That's the whole expensive part; everything below "
            f"is just fits over a cache."
        )
    mo.md(_msg)
    return


@app.cell(hide_code=True)
def _(POPS, mo):
    mo.md(
        f"""
        ## 1. Load a prebuilt cache

        Usually you just `read_f2` an existing cache. This one holds **{len(POPS)} populations**.
        Point `STEPPE_F2_DIR` at a bigger cache you built with `extract-f2` for more to model.
        """
    )
    return


@app.cell
def _(os, steppe):
    F2DIR = os.environ.get(
        "STEPPE_F2_DIR", os.path.expanduser("~/.local/share/steppe/example")
    )
    POPS = [ln.strip() for ln in open(os.path.join(F2DIR, "pops.txt")) if ln.strip()]
    f2 = steppe.read_f2(F2DIR, device=0)
    return F2DIR, POPS, f2


@app.cell
def _():
    TARGET = "Czechia_EBA_CordedWare"
    LEFT = ["Russia_Samara_EBA_Yamnaya", "Turkey_N"]
    RIGHT = ["Mbuti", "Han", "Papuan", "Karitiana", "Iran_GanjDareh_N", "Israel_Natufian"]
    return LEFT, RIGHT, TARGET


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 2. Fit a qpAdm model

        Can **Czechia_EBA_CordedWare** (Corded Ware) be explained as a mix of
        **Russia_Samara_EBA_Yamnaya** (Yamnaya steppe herders) and **Turkey_N** (Anatolian
        farmer)? This is the "massive migration from the steppe" of Haak et al. 2015 — the study
        this tool is named for. `res.weights` comes back as a DataFrame: the ancestry proportions,
        each with a standard error and a z-score.
        """
    )
    return


@app.cell
def _(LEFT, RIGHT, TARGET, f2, steppe):
    res = steppe.qpadm(f2, target=TARGET, left=LEFT, right=RIGHT)
    res.weights
    return (res,)


@app.cell(hide_code=True)
def _(plt, res):
    _w = res.weights
    _fig, _ax = plt.subplots(figsize=(5, 3))
    _ax.bar(_w["left"], _w["weight"], yerr=_w["se"], capsize=6, color=["#4C72B0", "#DD8452"])
    _ax.set_ylabel("ancestry proportion")
    _ax.set_ylim(0, 1)
    _ax.set_title(f"{res.target} ancestry (± se)")
    _fig
    return


@app.cell(hide_code=True)
def _(mo, res):
    mo.md(
        f"""
        The fit: **{res.weights.iloc[0]['left']} ≈ {res.weights.iloc[0]['weight']:.2f}**,
        **{res.weights.iloc[1]['left']} ≈ {res.weights.iloc[1]['weight']:.2f}**, model tail
        **p = {res.p:.3f}** (p > 0.05 → the model isn't rejected — a plausible fit).

        ## 3. Is the model any good? (`popdrop`)

        `res.popdrop` refits with each source dropped, so you can see which sources are
        load-bearing: drop one and if the fit collapses (tiny p), that source mattered.
        """
    )
    return


@app.cell
def _(res):
    res.popdrop
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 4. Compare many models at once (rotation)

        The real payoff: fit many candidate source-sets and get **one row per model**. Then filter
        to the feasible ones and sort by fit. (The pool below adapts to whatever's in your cache —
        a bigger cache gives a much richer rotation.)
        """
    )
    return


@app.cell
def _(POPS, RIGHT, TARGET, f2, itertools, steppe):
    _candidates = [
        "Russia_Samara_EBA_Yamnaya", "Turkey_N", "Serbia_IronGates_Mesolithic",
        "Czechia_EBA_CordedWare", "Russia_Saratov_Eneolithic_Khvalynsk",
        "Austria_N_LBK", "France_Yonne_N", "Ukraine_N",
    ]
    _pool = [p for p in _candidates if p in POPS]
    _models = [list(c) for k in (1, 2) for c in itertools.combinations(_pool, k)]
    rotation = steppe.qpadm_search(
        f2, target=TARGET, models=_models, right=RIGHT, as_dataframe=True
    )
    rotation[rotation["feasible"]].sort_values("p", ascending=False)
    return (rotation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## 5. f-statistics come back as tables too

        `steppe.f4(..., as_dataframe=True)` returns a DataFrame of f4 statistics — sort by `z` to
        surface the strongest signals. (A full billion-quartet sweep works the same way, just
        bigger.)
        """
    )
    return


@app.cell
def _(POPS, f2, steppe):
    _all = [
        ["Mbuti", "Han", "England_BellBeaker", "Czechia_EBA_CordedWare"],
        ["Mbuti", "Papuan", "England_BellBeaker", "Turkey_N"],
        ["Mbuti", "Han", "England_BellBeaker", "Russia_Samara_EBA_Yamnaya"],
        ["Mbuti", "Karitiana", "England_BellBeaker", "Serbia_IronGates_Mesolithic"],
    ]
    _quartets = [q for q in _all if all(p in POPS for p in q)]
    f4_table = steppe.f4(f2, _quartets, as_dataframe=True)
    f4_table.sort_values("z", key=abs, ascending=False)
    return (f4_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---
        That's the whole loop: **genotypes → `extract_f2` → an f2 cache → fits**, and every result
        is a DataFrame you can filter, sort, plot, join with metadata, and export. It matters most
        because Steppe produces *thousands* of these fast — a table is how you make sense of the
        flood.
        """
    )
    return


if __name__ == "__main__":
    app.run()
