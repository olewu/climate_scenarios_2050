"""Forecast at one site: mean and spread of each method, per window and scenario."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = {"M0": "0.3", "M1": "0.7", "M2a": "C1", "M2b": "C3", "M3a": "C2", "M3b": "C0"}


def plot_site(cfg, out, table, name, attrs):
    fig_dir = Path(cfg["paths"]["outputs"]["figures"]) / "forecast" / out.hash
    fig_dir.mkdir(parents=True, exist_ok=True)
    scenarios = list(table["scenario"].unique())
    windows = list(table["window"].unique())
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(max(7, len(windows) * 1.1), 2.6 * len(scenarios)), sharex=True, sharey=True, squeeze=False)
    for ax, scenario in zip(axes[:, 0], scenarios):
        sub = table[table["scenario"] == scenario]
        methods = list(sub["fc_method"].unique())
        offsets = np.linspace(-0.3, 0.3, len(methods))
        for off, method in zip(offsets, methods):
            row = sub[sub["fc_method"] == method].set_index("window").reindex(windows)
            ax.errorbar(np.arange(len(windows)) + off, row["mean"].values, yerr=row["spread"].values, fmt="o", ms=3, lw=1, color=COLORS.get(method, "k"), label=method)
        if "observed" in sub:
            obs = sub[sub["fc_method"] == methods[0]].set_index("window").reindex(windows)["observed"]
            ax.scatter(np.arange(len(windows)), obs.values, marker="_", s=200, color="k", label="observed", zorder=5)
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_ylabel(f"change [{attrs['units']}]")
        ax.set_title(scenario, fontsize=9)
    axes[0, 0].legend(ncol=7, fontsize=7)
    axes[-1, 0].set_xticks(np.arange(len(windows)))
    axes[-1, 0].set_xticklabels(windows, rotation=45)
    axes[-1, 0].set_xlabel(f"lead years after init {attrs['init_year']}, reference {attrs['reference_period']}")
    fig.savefig(fig_dir / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
