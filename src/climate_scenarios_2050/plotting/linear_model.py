"""
Maps of the linear model (hindcast fits, averaged over the init years):
- correlation of the predicted and observed internal target change given the observed index,
  predicted change = slope * (index window mean - index reference mean). Only the part explained by
  the index change; unlike `perfect_index` of the target evaluation, no reversion to the mean is included.
- target change for a typical index anomaly: slope * std of the window means of the observed internal
  training index (forced part fitted to all years; the leave-out internal window means of the
  hindcast cases are inflated), in units of the target data and, for relative targets, in % of the mean.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from climate_scenarios_2050.core.config import target_spec
from climate_scenarios_2050.linear_model.fit import sliding_means
from climate_scenarios_2050.preprocess import obs
from climate_scenarios_2050.split.run import SplitResults

WINDOWS = ["1-5", "1-10", "11-20", "1-20"]
UNITS = {"wsp100": "m/s"}   # files without a units attribute


def window_index_std(cfg, scenario, windows) -> xr.DataArray:
    """std of the sliding window means of the observed internal training index, per window"""
    split = SplitResults(cfg, scenario)
    name = cfg["obs"]["training_index"]
    internal = split.obs_internal(name, split.obs_coef(name, "coef_full")).dropna("year").values
    lengths = [int(w.split("-")[1]) - int(w.split("-")[0]) + 1 for w in windows]
    return xr.DataArray([np.nanstd(sliding_means(internal, n)) for n in lengths], dims="window", coords={"window": list(windows)})


def plot_linear_model(cfg, out, scenario, target_variable):
    fig_dir = Path(cfg["paths"]["outputs"]["figures"]) / "linear_model" / out.hash
    fig_dir.mkdir(parents=True, exist_ok=True)
    skill = xr.open_dataset(out.path("perfect_index_skill.nc"))
    params = xr.open_dataset(out.path("hindcast.nc"))
    wind = obs.open_obs_target(cfg, target_variable)
    units = wind.attrs.get("units") or UNITS.get(target_variable, "units of the target")
    index_std = window_index_std(cfg, scenario, params.window.values)
    change = params["slope"].mean("init_year") * index_std
    relative = change / wind.mean("year") * 100

    rows = [
        (skill["corr"], "corr given obs. index change", {"vmin": -1, "vmax": 1}, "correlation"),
        (change, "change per 1 std index", {"robust": True, "center": 0}, units),
    ]
    if target_spec(cfg, target_variable)["change"] == "relative":
        rows.append((relative, "change per 1 std index", {"robust": True, "center": 0}, "%"))
    fig, axes = plt.subplots(len(rows), len(WINDOWS), figsize=(14, 3 * len(rows)), subplot_kw={"projection": ccrs.PlateCarree()}, squeeze=False)
    for jj, (field, name, kwargs, label) in enumerate(rows):
        for ii, window in enumerate(WINDOWS):
            ax = axes[jj, ii]
            field.sel(window=window).plot(ax=ax, transform=ccrs.PlateCarree(), cmap="RdBu_r", add_labels=False, cbar_kwargs={"shrink": 0.7, "label": label}, **kwargs)
            ax.coastlines(lw=0.4)
            title = f"{name}, {window}"
            if jj > 0:
                title += f"\n(std of {window} index mean: {float(index_std.sel(window=window)):.2f})"
            ax.set_title(title, fontsize=8)
    fig.suptitle(f"Linear model {target_variable}, {scenario}: skill given the observed index change, and target change for a typical index anomaly")
    fig.savefig(fig_dir / f"linear_model_{target_variable}_{scenario}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
