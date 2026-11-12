"""Skill figures of the index hindcast evaluation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

COLORS = {"analogue": "C0", "cmip6_all": "k", "ar1_fair": "C1", "ar1_all": "C3", "sine_trend": "C2", "persistence": "C5"}


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


def plot_evaluation(cfg, out, scenario):
    fig_dir = Path(cfg["paths"]["outputs"]["figures"]) / "evaluate_nao" / out.hash
    fig_dir.mkdir(parents=True, exist_ok=True)
    cont = xr.open_dataset(out.path("continuous.nc"))
    win = xr.open_dataset(out.path("windows.nc"))
    null = xr.open_dataset(out.path("null.nc")) if out.exists("null.nc") else None

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    _lines(axes[0], cont, "corr", None if null is None else null["continuous_corr_p95"])
    axes[0].set_ylabel("correlation")
    _lines(axes[1], cont, "nrmse", None if null is None else null["continuous_nrmse_p95"])
    axes[1].set_ylabel("1 - RMSE/std")
    axes[0].legend(ncol=2, fontsize=7)
    fig.suptitle(f"Index hindcast skill, {scenario}, running means")
    fig.savefig(fig_dir / f"continuous_{scenario}.png", dpi=200, bbox_inches="tight")
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
