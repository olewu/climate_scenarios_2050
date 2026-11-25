"""
Forecast at one site: violins of the predictive distribution of each method, per window and scenario
(design of the old `make_vio_plot`). Each distribution is a Gaussian mixture (see
`forecast.site_components`): one Gaussian for M0/M1/M0c/M0t, the members' Gaussians for M2b/M3b, a weighted
kernel density of the member changes for M2a/M3a. Medians as horizontal bars, 5-95 % as thin lines,
the observed change (if the window lies in the record) dashed.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import ndtr

COLORS = {"M0": "0.35", "M1": "0.7", "M0c": "C5", "M0t": "C4", "M2a": "C1", "M2b": "C3", "M3a": "C2", "M3b": "C0"}


def mixture_density(mu, sigma, w, grid):
    z = (grid[None, :] - mu[:, None]) / sigma[:, None]
    return (w[:, None] * np.exp(-0.5 * z**2) / (sigma[:, None] * np.sqrt(2 * np.pi))).sum(axis=0)


def violin(ax, position, components, color, width, grid):
    mu, sigma, w = components
    density = mixture_density(mu, sigma, w, grid)
    cdf = (w[:, None] * ndtr((grid[None, :] - mu[:, None]) / sigma[:, None])).sum(axis=0)
    inside = (cdf > 0.001) & (cdf < 0.999)
    half = density / density.max() * width / 2
    ax.fill_betweenx(grid[inside], position - half[inside], position + half[inside], facecolor=color, alpha=0.6, lw=0)
    q05, q50, q95 = (np.interp(q, cdf, grid) for q in (0.05, 0.5, 0.95))
    ax.vlines(position, q05, q95, color=color, lw=0.8)
    ax.hlines(q50, position - width / 2, position + width / 2, color=color, lw=1.2)


def plot_site(cfg, out, table, components, name, attrs):
    fig_dir = Path(cfg["paths"]["outputs"]["figures"]) / "forecast" / out.hash
    fig_dir.mkdir(parents=True, exist_ok=True)
    scenarios = list(table["scenario"].unique())
    windows = list(table["window"].unique())
    methods = [m for m in COLORS if (table["fc_method"] == m).any() and table.loc[table["fc_method"] == m, "mean"].notnull().any()]
    x = np.arange(len(windows))
    offset = 0.8 / len(methods)
    width = 0.8 * offset

    lows = [(c[0] - 4 * c[1]).min() for c in components.values()]
    highs = [(c[0] + 4 * c[1]).max() for c in components.values()]
    grid = np.linspace(min(lows), max(highs), 1500)

    fig, axes = plt.subplots(len(scenarios), 1, figsize=(max(8, len(windows) * 1.2), 3.2 * len(scenarios)), sharex=True, sharey=True, squeeze=False)
    for ax, scenario in zip(axes[:, 0], scenarios):
        for jj, method in enumerate(methods):
            position = x - 0.4 + (jj + 0.5) * offset
            for ii, window in enumerate(windows):
                comp = components.get((scenario, window, method))
                if comp is not None:
                    violin(ax, position[ii], comp, COLORS[method], width, grid)
        sub = table[table["scenario"] == scenario]
        if "observed" in sub:
            observed = sub.groupby("window", sort=False)["observed"].first().reindex(windows)
            for ii, value in enumerate(observed.values):
                if np.isfinite(value):
                    ax.hlines(value, x[ii] - 0.45, x[ii] + 0.45, color="k", ls="dashed", lw=1)
        ax.axhline(0, color="grey", lw=0.5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel(f"change [{attrs['units']}]")
        ax.set_title(scenario, fontsize=9)
        # legend per scenario with its number of members
        for method in methods:
            sizes = [len(c[2]) for key, c in components.items() if key[0] == scenario and key[2] == method]
            count = f" ({max(sizes)})" if sizes and max(sizes) > 1 else ""
            ax.plot([], [], color=COLORS[method], lw=6, alpha=0.6, label=method + count)
        if "observed" in sub and sub["observed"].notnull().any():
            ax.plot([], [], color="k", ls="dashed", label="observed")
        ax.legend(ncol=len(methods) + 1, fontsize=7)
    axes[-1, 0].set_xticks(x)
    axes[-1, 0].set_xticklabels(windows, rotation=45)
    axes[-1, 0].set_xlabel(f"lead years after init {attrs['init_year']}, reference {attrs['reference_period']}")
    fig.suptitle(name.replace("_", " "), fontsize=10)
    fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
