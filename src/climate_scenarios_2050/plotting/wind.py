"""Skill maps of the wind methods."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

WINDOWS = ["1-5", "6-10", "1-10", "11-20", "1-20", "21-30"]


def plot_wind_evaluation(cfg, out, scenario, wind_variable):
    fig_dir = Path(cfg["paths"]["outputs"]["figures"]) / "evaluate_wind" / out.hash
    fig_dir.mkdir(parents=True, exist_ok=True)
    scores = xr.open_dataset(out.path("scores.nc"))
    for target in scores.target.values:
        for var, vmin, vmax in [("corr", -1, 1), ("crpss", -0.5, 0.5)]:
            sub = scores[var].sel(target=target)
            methods = [m for m in ["M0", "M2a", "M3a", "M2b", "M3b", "perfect_nao"] if m in sub.fc_method.values and bool(sub.sel(fc_method=m).notnull().any())]
            fig, axes = plt.subplots(len(methods), len(WINDOWS), figsize=(2.6 * len(WINDOWS), 2.0 * len(methods)), subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
            for jj, method in enumerate(methods):
                for ii, window in enumerate(WINDOWS):
                    ax = axes[jj, ii]
                    mesh = sub.sel(fc_method=method, window=window).plot(ax=ax, transform=ccrs.PlateCarree(), cmap="RdBu_r", vmin=vmin, vmax=vmax, add_labels=False, add_colorbar=False)
                    ax.coastlines(lw=0.3)
                    ax.set_title(f"{method} {window}", fontsize=8)
            fig.colorbar(mesh, ax=axes, shrink=0.6, label=var)
            fig.savefig(fig_dir / f"{var}_{target}_{wind_variable}_{scenario}.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
