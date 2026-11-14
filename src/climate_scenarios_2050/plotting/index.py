"""
Figures of the index hindcast evaluation:
- `skill_corr_*`, `skill_rmsess_*`: continuous skill per lead time in the design of the old
  `evaluate_full_synth.py` (analogue, AR(1) fair/full, synthetic 95th percentile, DCPP reference)
- `continuous_all_*`: the same scores for all products
- `hindcast_*`: visual verification, ensemble-mean hindcast by initialization and valid year, with the
  observations and fixed-lead hindcasts below
- `windows_*`: skill of the window changes
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import proplot as pplt

COLORS = {"analogue": "C0", "cmip6_all": "k", "ar1_fair": "C1", "ar1_all": "C3", "sine_trend": "C2", "persistence": "C5"}

# reference skill (e.g. the DCPP multi-model NAO correlation for lead years 2-9) from
# `predictor.reference_skill`, shown as a bar at its centre lead time
FIXED_LEADS = {5: "solid", 15: "dashed", 25: "dotted"}


def _lines(ax, ds, measure, null=None):
    for name in ds["product"].values:
        sub = ds.sel(product=name)
        color = COLORS.get(str(name), "C4")
        ax.plot(sub.lead, sub[f"{measure}_score"], color=color, label=str(name))
        ax.fill_between(sub.lead, sub[f"{measure}_ci_low"], sub[f"{measure}_ci_high"], color=color, alpha=0.12, lw=0)
    if null is not None:
        ax.plot(null.lead, null, color="grey", ls="dotted", label="synthetic p95")
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_xlabel("lead time [years]")


def legacy_skill_figure(cont, null, measure, index_name, path, reference=None):
    """skill per lead time in the old design"""
    title, ylabel, ylim, legend_loc = {
        "corr": ("Correlation Skill", "Correlation", (-0.4, 0.9), 3),
        "nrmse": ("RMSE Skill", "RMSESS", (-0.5, 0.5), 1),
    }[measure]
    leads = cont.lead.values
    fig = pplt.figure(suptitle=f"{index_name} {title}", figsize=[5, 3])
    ax = fig.add_subplot(xlabel="lead time [a]", ylabel=ylabel)
    ax.hlines(0, leads[0], leads[-1], color="grey", linestyle="dashed")
    sel = lambda name, key: cont[f"{measure}_{key}"].sel(product=name).values
    if "ar1_fair" in cont["product"].values:
        ax.plot(leads, sel("ar1_fair", "score"), color="C1", label="AR(1) fair")
        ax.fill_between(leads, sel("ar1_fair", "ci_low"), sel("ar1_fair", "ci_high"), alpha=0.2, color="C1", label="CI AR(1)")
    if "ar1_all" in cont["product"].values:
        ax.plot(leads, sel("ar1_all", "score"), color="C1", label="AR(1) full", ls="dashed", lw=0.5)
    ax.plot(leads, sel("analogue", "score"), color="C0", label="Analogue")
    ax.fill_between(leads, sel("analogue", "ci_low"), sel("analogue", "ci_high"), alpha=0.2, color="C0", label="CI analogue")
    if null is not None:
        ax.plot(null.lead.values, null[f"continuous_{measure}_p95"].values, color="k", label="random 95th", ls="dotted")
    if measure == "corr" and reference:
        ref = reference
        ax.plot([ref["lead"], ref["lead"]], [ref["low"], ref["high"]], color="k", marker="_")
        ax.annotate(text="  " + ref["label"], xy=(ref["lead"], ref["high"] + 0.01), ha="left", va="center")
    ax.legend(loc=legend_loc, ncol=3)
    ax.format(ylim=list(ylim), xlim=[leads[0], leads[-1]])
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def hindcast_figure(ov: xr.Dataset, scenario, path, lock_year=1985):
    """ensemble-mean hindcast by initialization and valid year, observations and fixed-lead hindcasts below"""
    index_name = ov.attrs["index"].upper()
    hc = ov["analogue_mean"].where(ov.lead > 0)
    inits = hc.init_year.values
    years = np.arange(inits.min() + 1, inits.max() + int(hc.lead.max()) + 1)
    # (init_year, year)
    grid = np.full((len(inits), len(years)), np.nan)
    leads_grid = years[None, :] - inits[:, None]
    for ii, init in enumerate(inits):
        valid = hc.sel(init_year=init).dropna("lead")
        grid[ii, np.searchsorted(years, init + valid.lead.values)] = valid.values
    leads_grid = np.where(leads_grid > 0, leads_grid, np.nan)
    obs = ov["observed"].dropna("year")
    limit = float(np.nanquantile(np.abs(grid), 0.98))

    array = [[1], [1], [1], [2]]
    fig, axs = pplt.subplots(array, figwidth=5, figheight=4.8, suptitle=f"{index_name} ensemble mean hindcast ({scenario})", sharex=True)
    axs.format(abc=True)
    cf = axs[0].pcolormesh(years, inits, grid, vmin=-limit, vmax=limit, cmap="RdBu_r", extend="both")
    axs[0].set_facecolor("lightgrey")
    in_obs = years <= obs.year.max().values
    cs = axs[0].contour(years[in_obs], inits, leads_grid[:, in_obs], levels=[5, 15, 25, 35], colors="k", linestyles="dotted", linewidths=0.6)
    axs[0].clabel(cs, fmt="%d", fontsize=6)
    axs[0].colorbar(cf, loc="r", label="")
    axs[0].format(xlabel="Year", ylabel="Hindcast Initialization Year", xlim=[inits.min(), int(obs.year.max())], ylim=[inits.min(), inits.max()])

    window = ov.attrs.get("matching_window", "all")
    if window != "all":
        window = int(window)
        lock_year = min(max(lock_year, int(inits.min()) + window), int(inits.max()))
        axs[0].annotate(text="", xy=(lock_year, lock_year), xytext=(lock_year - window, lock_year), arrowprops=dict(arrowstyle="<->"))
        axs[0].text(lock_year - window // 2, lock_year, f"matching window ({window} years)", ha="center", va="bottom", fontsize=7,
                    bbox={"fc": "lightgrey", "ec": "none", "pad": 0})

    axs[1].plot(obs.year.values, obs.values, label=ov.attrs.get("observations", "observed"))
    for lead, ls in FIXED_LEADS.items():
        if lead in hc.lead.values:
            fixed = hc.sel(lead=lead)
            axs[1].plot(fixed.init_year.values + lead, fixed.values, ls=ls, lw=0.75, color="k", label=f"LT{lead}Y")
    axs[1].axhline(0, color="grey", lw=0.5)
    axs[1].format(ylabel=f"{index_name} index", xlim=[inits.min(), int(obs.year.max())])
    axs[1].legend(loc="b", fontsize=8, ncols=4, frame=False)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_evaluation(cfg, out, scenario):
    fig_dir = Path(cfg["paths"]["outputs"]["figures"]) / "evaluate_index" / out.hash
    fig_dir.mkdir(parents=True, exist_ok=True)
    cont = xr.open_dataset(out.path("continuous.nc"))
    win = xr.open_dataset(out.path("windows.nc"))
    null = xr.open_dataset(out.path("null.nc")) if out.exists("null.nc") else None
    index_name = cfg["predictor"]["name"].upper()

    for measure, tag in [("corr", "corr"), ("nrmse", "rmsess")]:
        legacy_skill_figure(cont, null, measure, index_name, fig_dir / f"skill_{tag}_{scenario}.png", cfg["predictor"].get("reference_skill"))
    if out.exists("hindcast_overview.nc"):
        hindcast_figure(xr.open_dataset(out.path("hindcast_overview.nc")), scenario, fig_dir / f"hindcast_{scenario}.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    _lines(axes[0], cont, "corr", None if null is None else null["continuous_corr_p95"])
    axes[0].set_ylabel("correlation")
    _lines(axes[1], cont, "nrmse", None if null is None else null["continuous_nrmse_p95"])
    axes[1].set_ylabel("1 - RMSE/std")
    axes[0].legend(ncol=2, fontsize=7)
    fig.suptitle(f"Index hindcast skill, {scenario}, running means")
    fig.savefig(fig_dir / f"continuous_all_{scenario}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    labels = [str(w) for w in win.window.values]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.5), sharex=True)
    for ax, var, title in zip(axes, ["corr", "nrmse", "crpss"], ["correlation", "1 - RMSE/std", "CRPSS (reference: no change)"]):
        for name in win["product"].values:
            if var in win and bool(win[var].sel(product=name).notnull().any()):
                ax.plot(labels, win[var].sel(product=name), marker="o", ms=3, color=COLORS.get(str(name), "C4"), label=str(name))
        if null is not None and f"window_{var}_p95" in null:
            ax.plot(labels, null[f"window_{var}_p95"], color="grey", ls="dotted", label="synthetic p95")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=60)
    axes[0].legend(ncol=2, fontsize=7)
    fig.suptitle(f"Index hindcast skill of window changes, {scenario}")
    fig.savefig(fig_dir / f"windows_{scenario}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
