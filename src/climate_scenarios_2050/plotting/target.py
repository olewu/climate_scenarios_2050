"""Skill maps of the target methods."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

WINDOWS = ["1-5", "6-10", "1-10", "11-20", "1-20", "21-30"]
METHODS = ["M0", "M0c", "M0t", "M2a", "M3a", "M2b", "M3b", "perfect_index"]
# score map -> statistic whose significance (beyond M0) is stippled on it
STIPPLE = {"corr": "dcorr", "crpss": "dcrpss"}
LIMITS = {"corr": 1, "crpss": 0.5, "dcrpss": 0.3, "dcorr": 0.5, "corr_anom": 1}
# colour bar labels of the statistics beyond M0 (m: the method of the row)
LABELS = {
    "dcrpss": "dcrpss = CRPSS(m) - CRPSS(M0)",
    "dcorr": "dcorr = corr(m, obs) - corr(M0, obs)",
    "corr_anom": "corr_anom = corr(m - M0, obs - M0)",
}


def stipple(ax, sig: xr.DataArray):
    """hatch the grid points significant after the FDR"""
    if not bool(sig.any()):
        return
    ax.contourf(sig.longitude.values, sig.latitude.values, sig.values.astype(float), levels=[0.5, 1.5], colors="none", hatches=["...."], transform=ccrs.PlateCarree())


def field_label(sig: xr.Dataset, statistic, target, method, window) -> str:
    sub = sig.sel(statistic=statistic, target=target, fc_method=method, window=window)
    if not np.isfinite(sub["field_mean"]):
        return ""
    return f"\n{float(sub['field_mean']):+.2f} (p={float(sub['p_field_mean']):.3f}), {100 * float(sub['area_fraction_positive']):.0f}% (p={float(sub['p_area_fraction']):.3f})"


def map_grid(da: xr.DataArray, methods, limit, label, path, sig=None, statistic=None, target=None):
    fig, axes = plt.subplots(len(methods), len(WINDOWS), figsize=(2.6 * len(WINDOWS), 2.2 * len(methods)), subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    for jj, method in enumerate(methods):
        for ii, window in enumerate(WINDOWS):
            ax = axes[jj, ii]
            mesh = da.sel(fc_method=method, window=window).plot(ax=ax, transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=-limit, vmax=limit, add_labels=False, add_colorbar=False)
            ax.coastlines(lw=0.3)
            title = f"{method} {window}"
            if sig is not None and method in sig.fc_method.values and method != "M0":
                stipple(ax, sig["significant"].sel(statistic=statistic, target=target, fc_method=method, window=window).astype(bool))
                title += field_label(sig, statistic, target, method, window)
            ax.set_title(title, fontsize=6 if sig is not None else 8)
    fig.colorbar(mesh, ax=axes, shrink=0.6, label=label)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_target_evaluation(cfg, out, scenario, target_variable):
    fig_dir = Path(cfg["paths"]["outputs"]["figures"]) / "evaluate_target" / out.hash
    fig_dir.mkdir(parents=True, exist_ok=True)
    scores = xr.open_dataset(out.path("scores.nc"))
    sig = xr.open_dataset(out.path("significance.nc")) if out.exists("significance.nc") else None
    note = "" if sig is None else f" (dots: beats M0, FDR q={sig.attrs['fdr_q']:.2f}; titles: area mean (field p), area beating M0 (p))"
    for target in scores.target.values:
        # M2a on the internal target is zero up to rounding (older outputs still contain it)
        hidden = {"M2a"} if target == "internal" else set()
        for var in ["corr", "crpss"]:
            sub = scores[var].sel(target=target)
            methods = [m for m in METHODS if m not in hidden and m in sub.fc_method.values and bool(sub.sel(fc_method=m).notnull().any())]
            map_grid(sub, methods, LIMITS[var], var + note, fig_dir / f"{var}_{target}_{target_variable}_{scenario}.png", sig, STIPPLE[var], target)
        if sig is None:
            continue
        for statistic in sig.statistic.values:
            sub = sig["value"].sel(statistic=statistic, target=target)
            methods = [m for m in METHODS[1:] + ["M1"] if m not in hidden and m in sub.fc_method.values and bool(sub.sel(fc_method=m).notnull().any())]
            map_grid(sub, methods, LIMITS[str(statistic)], LABELS.get(str(statistic), f"{statistic} (beyond M0)") + note, fig_dir / f"{statistic}_{target}_{target_variable}_{scenario}.png", sig, str(statistic), target)
