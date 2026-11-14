"""
Analogue selection: compare the filtered observed index with the filtered index of each CMIP6
member over a matching window that ends at the init year, using only what is known at init.

Replaces `ens_subsample`/`ens_shift` of the old `generate_NAO_hindcast.py`, with these fixes:
- every member is centred by its own time mean over the window, like the observations. The old
  code subtracted the single-model ensemble mean per year, which made single-member models all
  zeros (correlation NaN) and reduced members of small ensembles to their differences
- NaN scores are dropped before ranking (a descending sort puts NaN first, so all single-member
  models used to enter every analogue set)
- `model_filter_at_init` makes the treatment of the model series explicit. `full` (default, as before)
  applies the centred filter over the whole model series: no observed future is used, since the
  runs' future exists at forecast time too, and members are selected whose next years stay close to the
  recent observed state. `cut` filters the model series exactly like the observations (no NAO skill).
- the window has exactly `window` years, ending at the init year (the old one had window + 1)
"""

import numpy as np

from climate_scenarios_2050.core.filters import bandpass, lowpass


#---------------filters--------------#
def running_mean(X: np.ndarray, length: int, min_periods: int) -> np.ndarray:
    """centred running mean along the last axis, using the available values in each window"""
    X = np.atleast_2d(X)
    valid = np.isfinite(X)
    csum = np.concatenate([np.zeros((*X.shape[:-1], 1)), np.cumsum(np.where(valid, X, 0.0), axis=-1)], axis=-1)
    ccnt = np.concatenate([np.zeros((*X.shape[:-1], 1)), np.cumsum(valid, axis=-1)], axis=-1)
    n = X.shape[-1]
    lo = np.clip(np.arange(n) - length // 2, 0, n)
    hi = np.clip(np.arange(n) + (length - 1) // 2 + 1, 0, n)
    count = ccnt[..., hi] - ccnt[..., lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (csum[..., hi] - csum[..., lo]) / count
    out[count < min_periods] = np.nan
    return out


def apply_filter(X: np.ndarray, filter_cfg: dict) -> np.ndarray:
    """apply the configured filter along the last axis of a (n, time) array"""
    X = np.atleast_2d(X)
    kind = filter_cfg["type"]
    if kind == "running_mean":
        length = filter_cfg["length"]
        return running_mean(X, length, max(1, int(length * filter_cfg.get("min_periods_frac", 1.0))))
    if kind == "lowpass":
        return np.apply_along_axis(lowpass, -1, X, filter_cfg["cutoff_periods"][0], filter_cfg.get("order", 4))
    if kind == "bandpass":
        short, long = filter_cfg["cutoff_periods"]
        return np.apply_along_axis(bandpass, -1, X, short, long, filter_cfg.get("order", 4))
    raise ValueError(f"unknown filter type {kind}")


#---------------scores--------------#
def scores(obs_win: np.ndarray, model_win: np.ndarray, measure: str) -> np.ndarray:
    """
    score of each member against the observations over the window. Both are centred over the
    years a member has data for (see `min_window_coverage` for how many years are required).
    """
    valid = np.isfinite(model_win)
    with np.errstate(invalid="ignore", divide="ignore"):
        n = valid.sum(axis=-1, keepdims=True)
        model_anom = np.where(valid, model_win, 0.0)
        model_anom = np.where(valid, model_anom - model_anom.sum(axis=-1, keepdims=True) / n, 0.0)
        obs_masked = np.where(valid, obs_win, 0.0)
        obs_anom = np.where(valid, obs_masked - obs_masked.sum(axis=-1, keepdims=True) / n, 0.0)
        if measure == "RMSE":
            return np.sqrt(((model_anom - obs_anom) ** 2).sum(axis=-1) / n[..., 0])
        if measure == "CORR":
            return (model_anom * obs_anom).sum(axis=-1) / np.sqrt((model_anom**2).sum(axis=-1) * (obs_anom**2).sum(axis=-1))
    raise ValueError(f"unknown measure {measure}")


def rank(score: np.ndarray, measure: str) -> np.ndarray:
    """indices of the valid members from best to worst"""
    valid = np.flatnonzero(np.isfinite(score))
    order = np.argsort(score[valid] if measure == "RMSE" else -score[valid], kind="stable")
    return valid[order]


#---------------selection--------------#
def shifted(model: np.ndarray, lag: int) -> np.ndarray:
    """value at year label t = member value of year t + lag"""
    if lag == 0:
        return model
    out = np.full_like(model, np.nan)
    if lag > 0:
        out[..., :-lag] = model[..., lag:]
    else:
        out[..., -lag:] = model[..., :lag]
    return out


def window_mask(years: np.ndarray, init: int, window) -> np.ndarray:
    if window == "all":
        return years <= init
    return (years > init - window) & (years <= init)


def member_scores_for_lag(model, years, obs_win, init, cfg, lag, measure, mask):
    """filtered members (cut at init if configured) and their score at one lag"""
    model_lag = shifted(model, lag)
    known = years <= init
    # members need data over the window (the filter would otherwise fill gaps with partial means)
    coverage = np.isfinite(model_lag[:, mask]).mean(axis=1)
    if cfg["model_filter_at_init"] == "cut":
        cut = model_lag.copy()
        cut[:, ~known] = np.nan
        filtered = apply_filter(cut, cfg["filter"])
    elif cfg["model_filter_at_init"] == "full":
        filtered = shifted(apply_filter(model, cfg["filter"]), lag)
    else:
        raise ValueError(f"unknown model_filter_at_init {cfg['model_filter_at_init']}")
    score = scores(obs_win, filtered[:, mask], measure)
    score[coverage < cfg["min_window_coverage"]] = np.nan
    return filtered, score


def combine(scores_k: np.ndarray, weights: np.ndarray, measure: str) -> np.ndarray:
    """
    joint score of several indices (index first): weighted mean of the correlations, or the root of the
    weighted mean of the squared errors (RMSE); NaN if any index has no score
    """
    w = weights.reshape(-1, *([1] * (scores_k.ndim - 1)))
    if measure == "RMSE":
        return np.sqrt((w * scores_k**2).sum(axis=0))
    return (w * scores_k).sum(axis=0)


def select_members(obs: np.ndarray, obs_years: np.ndarray, model: np.ndarray, years: np.ndarray, init: int, cfg: dict, weights=None) -> dict:
    """
    the configured selection for one init year; `obs` and `model` are unfiltered internal
    variability, `model` has dims (member, year). Several indices: `obs` (index, year), `model`
    (index, member, year) and `weights` (index), scores combined with `combine` (the same lag for all
    indices of a member). Returns member indices (best first), their score and lag.
    """
    obs = np.atleast_2d(obs)
    model = model[None] if model.ndim == 2 else model
    n_index = obs.shape[0]
    weights = np.full(n_index, 1 / n_index) if weights is None else np.asarray(weights, dtype=float) / np.sum(weights)
    window = cfg["window"]
    known = obs_years <= init
    obs_series = np.full((n_index, len(years)), np.nan)
    in_obs = np.isin(years, obs_years[known])
    for k in range(n_index):
        obs_filtered = apply_filter(obs[k, known], cfg["filter"])[0]
        obs_series[k, in_obs] = obs_filtered[np.isin(obs_years[known], years)]
    mask = window_mask(years, init, window) & in_obs
    if window != "all" and mask.sum() < window:
        raise ValueError(f"init {init}: matching window not covered by the observations")
    # years the filter cannot provide with the requested number of values (for any index)
    mask = mask & np.isfinite(obs_series).all(axis=0)
    obs_win = obs_series[:, mask]

    selection, measure = cfg["selection"], cfg["measure"]
    first_measure = "CORR" if measure == "two_stage" else measure
    n_members = model.shape[1]

    def joint(lag, measure_, rows=slice(None)):
        per_index = [member_scores_for_lag(model[k, rows], years, obs_win[k], init, cfg, lag, measure_, mask)[1] for k in range(n_index)]
        return combine(np.stack(per_index), weights, measure_)

    if selection == "subsample":
        best = joint(0, first_measure)
        best_lag = np.zeros(n_members, dtype=int)
    elif selection in ["shift", "shift_subsample"]:
        lags = np.arange(-cfg["max_lag"], cfg["max_lag"] + 1)
        all_scores = np.stack([joint(int(lag), first_measure) for lag in lags])
        filled = np.where(np.isfinite(all_scores), all_scores, np.inf if first_measure == "RMSE" else -np.inf)
        ilag = filled.argmin(axis=0) if first_measure == "RMSE" else filled.argmax(axis=0)
        best = all_scores[ilag, np.arange(n_members)]
        best_lag = lags[ilag]
    else:
        raise ValueError(f"unknown selection {selection}")

    order = rank(best, first_measure)
    if selection != "shift":
        M = cfg["M"]
        if measure == "two_stage":
            # keep the best by correlation (phase), then the best of those by RMSE (amplitude)
            order = order[: int(M / cfg["two_stage_fraction"])]
            amp = np.array([joint(int(best_lag[i]), "RMSE", [i])[0] for i in order])
            order = order[rank(amp, "RMSE")][:M]
        else:
            order = order[:M]
    return {"member": order, "score": best[order], "lag": best_lag[order], "n_valid": int(np.isfinite(best).sum())}


def align_leads(model: np.ndarray, years: np.ndarray, init: int, leads: np.ndarray, lags=None) -> np.ndarray:
    """(n, lead) values of member(s) at years init + lead (+ lag for shifted members)"""
    n = model.shape[0]
    lags = np.zeros(n, dtype=int) if lags is None else np.asarray(lags)
    pos = (init + leads)[None, :] + lags[:, None] - years[0]
    valid = (pos >= 0) & (pos < len(years))
    out = np.full((n, len(leads)), np.nan)
    rows = np.arange(n)[:, None].repeat(len(leads), axis=1)
    out[valid] = model[rows[valid], pos[valid]]
    return out
