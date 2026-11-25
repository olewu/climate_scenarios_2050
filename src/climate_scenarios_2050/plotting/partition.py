"""Variance partition figures: fractions (stacked) and standard deviations per source, one row per window length."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

COLORS = {"internal": "#1f77b4", "model": "#ff7f0e", "scenario": "#2ca02c", "total": "#000000"}
LABELS = {"internal": "internal variability", "model": "model uncertainty", "scenario": "scenario uncertainty"}


def plot_partition(cfg, out):
    ds = xr.open_dataset(out.path("partition.nc"))
    fig_dir = Path(cfg["paths"]["outputs"]["figures"]) / "partition" / out.hash
    fig_dir.mkdir(parents=True, exist_ok=True)
    name = ds.attrs.get("name", "")
    units = ds.attrs.get("units", "")
    lengths = ds.window_length.values
    fig, axes = plt.subplots(len(lengths), 2, figsize=(11, 3.1 * len(lengths)), squeeze=False)
    for j, length in enumerate(lengths):
        sub = ds.sel(window_length=length)
        x = sub.centre_year.values
        ok = np.isfinite(sub["total"].values)
        ax = axes[j, 0]
        ax.stackplot(x[ok], *[sub["fraction"].sel(source=s).values[ok] for s in ds.source.values],
                     labels=[LABELS[str(s)] for s in ds.source.values], colors=[COLORS[str(s)] for s in ds.source.values], alpha=0.85)
        ax.set_ylim(0, 1)
        ax.set_ylabel("fraction of variance")
        ax.set_title(f"{name}: {length}-year window mean change", fontsize=10)
        ax = axes[j, 1]
        for s in ds.source.values:
            ax.plot(x, np.sqrt(sub["variance"].sel(source=s).values), color=COLORS[str(s)], label=LABELS[str(s)])
        ax.plot(x, np.sqrt(sub["total"].values), color=COLORS["total"], lw=1.2, ls="--", label="total")
        ax.set_ylabel(f"std [{units}]")
        ax.set_ylim(bottom=0)
        ax.set_title(f"reference {ds.attrs.get('reference_period', '')}", fontsize=10)
        for a in axes[j]:
            a.axvline(2050, color="0.5", lw=0.8, ls=":")
            a.set_xlim(x[ok].min() if ok.any() else x.min(), x[ok].max() if ok.any() else x.max())
            a.grid(alpha=0.3)
    axes[0, 0].legend(loc="lower left", fontsize=8)
    axes[0, 1].legend(loc="upper left", fontsize=8)
    for a in axes[-1]:
        a.set_xlabel("centre year of the window")
    fig.suptitle(f"Variance partition of the projected change, {name} (scenarios: {', '.join(map(str, ds.scenario.values))})", fontsize=11)
    fig.tight_layout()
    path = fig_dir / f"partition_{name.replace(' ', '_')}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
