"""Maps of the linear model: slope and skill given the observed index."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import xarray as xr

WINDOWS = ["1-5", "1-10", "11-20", "1-20"]


def plot_linear_model(cfg, out, scenario, wind_variable):
    fig_dir = Path(cfg["paths"]["outputs"]["figures"]) / "linear_model" / out.hash
    fig_dir.mkdir(parents=True, exist_ok=True)
    skill = xr.open_dataset(out.path("perfect_index_skill.nc"))
    slope = xr.open_dataset(out.path("hindcast.nc"))["slope"].mean("init_year")
    rows = [(skill["corr"], "correlation", "RdBu_r", -1, 1), (slope, "mean slope per index unit", "RdBu_r", None, None)]
    fig, axes = plt.subplots(2, len(WINDOWS), figsize=(14, 6), subplot_kw={"projection": ccrs.PlateCarree()})
    for jj, (field, name, cmap, vmin, vmax) in enumerate(rows):
        for ii, window in enumerate(WINDOWS):
            ax = axes[jj, ii]
            kwargs = {"vmin": vmin, "vmax": vmax} if vmin is not None else {"robust": True, "center": 0}
            field.sel(window=window).plot(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap, add_labels=False, cbar_kwargs={"shrink": 0.7}, **kwargs)
            ax.coastlines(lw=0.4)
            ax.set_title(f"{name}, {window}", fontsize=9)
    fig.suptitle(f"Linear model {wind_variable}, {scenario}: skill given the observed index")
    fig.savefig(fig_dir / f"linear_model_{wind_variable}_{scenario}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
