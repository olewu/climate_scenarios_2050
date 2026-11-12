"""Optional restriction of the analogue pool to a set of models (from the old `utils.subsample`)."""

import pickle
from pathlib import Path

import numpy as np
from scipy import signal

SMITH_BEST = ["UKESM1-0-LL", "KACE-1-0-G", "MPI-ESM1-2-LR", "MRI-ESM2-0", "MIROC-ES2L", "HadGEM3-GC31-LL", "CanESM5-CanOE", "CanESM5-1", "CanESM5", "CESM2-WACCM", "CESM2", "ACCESS-CM2", "GISS-E2-1-G", "NorESM2-LM"]
SMITH_INTERMEDIATE = ["GFDL-ESM4", "ACCESS-ESM1-5", "MIROC6"]
SMITH_WORST = ["CNRM-ESM2-1", "CNRM-CM6-1", "EC-Earth3-Veg-LR", "EC-Earth3-Veg", "EC-Earth3", "IPSL-CM6A-LR", "GISS-E2-2-G"]


def lowfreq_variance_ratio(ts, cutoff, fs=1):
    """ratio of the low-frequency (f <= cutoff) to the total variance"""
    f, psd = signal.welch(ts, fs=fs)
    return np.trapz(psd[f <= cutoff], f[f <= cutoff]) / np.trapz(psd, f)


def selected_models(name, models, internal=None, obs=None, subsampling_dir=None) -> list:
    """model names of a pre-selection; `internal` (with a `model` coordinate) and `obs` for varsplit"""
    models = list(np.unique(models))
    if name == "Smith_incl_best":
        return [m for m in models if m in SMITH_BEST]
    if name == "Smith_incl_best_plus":
        return [m for m in models if m in SMITH_BEST + SMITH_INTERMEDIATE]
    if name == "Smith_excl_worst":
        return [m for m in models if m not in SMITH_WORST]
    if name.startswith("NAO_AR"):
        with open(Path(subsampling_dir) / f"{name}_subsampling.pkl", "rb") as f:
            return [m for m in pickle.load(f) if m in models]
    if name.startswith("varsplit_"):
        # keep models whose low-frequency variance ratio is within 0.1 of the observed one
        cutoff = 1 / int(name.split("_")[1].rstrip("y"))
        ratio_obs = lowfreq_variance_ratio(obs.dropna("year").values, cutoff)
        selected = []
        for model in models:
            sub = internal.isel(member=(internal["model"] == model).values)
            ratios = [lowfreq_variance_ratio(s.dropna("year").values, cutoff) for s in sub.transpose("member", "year")]
            if abs(ratio_obs - np.mean(ratios)) < 0.1:
                selected.append(model)
        return selected
    raise ValueError(f"unknown preselection {name}")
