"""
Verification scores for hindcasts given per sample (sample_dim, default init_year).
All scores use the same samples: pairs with a missing forecast or observation are removed first.

Event thresholds are one fixed value per window, the observed quantile over all samples. Thresholds
leaving out each sample's neighbours do not work for these slowly varying series: removing the
neighbours moves the threshold away from the sample, so the threshold reveals on which side the
sample lies, and even a forecast without any skill gets a high Brier skill score.
The spread of the Gaussian CRPS reference is a leave-out estimate (overlapping samples removed).
"""

import numpy as np
import xarray as xr
from scipy.special import ndtr


def common_mask(fc_mean, obs):
    return fc_mean.notnull() & obs.notnull()


def deterministic(fc_mean, obs, dim="init_year") -> xr.Dataset:
    mask = common_mask(fc_mean, obs)
    f, o = fc_mean.where(mask), obs.where(mask)
    rmse = np.sqrt(((f - o) ** 2).mean(dim))
    return xr.Dataset({
        "corr": xr.corr(f, o, dim=dim),
        "rmse": rmse,
        "nrmse": 1 - rmse / o.std(dim),
        "sign_agreement": ((np.sign(f) == np.sign(o)).where(mask)).mean(dim),
        "n_samples": mask.sum(dim),
    })


def crps_gaussian(mu, sigma, obs):
    z = (obs - mu) / sigma
    pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
    return sigma * (z * (2 * xr.apply_ufunc(ndtr, z) - 1) + 2 * pdf - 1 / np.sqrt(np.pi))


def _crps_ensemble(fc, obs):
    """fc: (..., member), obs: (...); NaN members are ignored"""
    fc = np.sort(fc, axis=-1)  # NaN last
    valid = np.isfinite(fc)
    n = valid.sum(axis=-1)
    term1 = np.nansum(np.abs(fc - obs[..., None]), axis=-1) / n
    rank = np.cumsum(valid, axis=-1)
    term2 = np.nansum((2 * rank - n[..., None] - 1) * fc, axis=-1) / n**2
    return term1 - term2


def crps_ensemble(fc, obs, member_dim="member"):
    return xr.apply_ufunc(_crps_ensemble, fc, obs, input_core_dims=[[member_dim], []], vectorize=False, dask="forbidden")


def separated(init_years, min_distance) -> np.ndarray:
    """True where two samples are at least `min_distance` apart, shape (n, n)"""
    init_years = np.asarray(init_years)
    return np.abs(init_years[:, None] - init_years[None, :]) >= min_distance


def leave_out_stats(obs, min_distance, dim="init_year"):
    obs = obs.transpose(dim, ...)
    keep = separated(obs[dim].values, min_distance)
    mean = xr.concat([obs.isel({dim: keep[ii]}).mean(dim) for ii in range(obs.sizes[dim])], dim=obs[dim])
    std = xr.concat([obs.isel({dim: keep[ii]}).std(dim) for ii in range(obs.sizes[dim])], dim=obs[dim])
    return mean, std


def probabilistic(fc, obs, min_distance, quantiles=(0.5, 0.9), member_dim="member", dim="init_year", reference_mean=None) -> xr.Dataset:
    """
    CRPS and its skill against a Gaussian reference, Brier score for exceeding the observed
    quantiles and its skill against the climatological probability. Reference mean: leave-out
    mean of the observations unless given (e.g. 0 for persistence of the reference period).
    """
    mask = fc.notnull().any(member_dim) & obs.notnull()
    fc = fc.where(mask)
    obs_m = obs.where(mask)

    ref_mean, ref_std = leave_out_stats(obs, min_distance, dim)
    if reference_mean is not None:
        # spread around the prescribed reference mean
        ref_mean = xr.full_like(ref_mean, reference_mean)
    crps = crps_ensemble(fc, obs_m.fillna(0), member_dim).where(mask).mean(dim)
    crps_ref = crps_gaussian(ref_mean, ref_std, obs).where(mask).mean(dim)

    thresholds = obs_m.quantile(list(quantiles), dim=dim, skipna=True)
    prob = (fc > thresholds).where(fc.notnull()).mean(member_dim)
    event = (obs > thresholds).astype(float)
    brier = ((prob - event) ** 2).where(mask).mean(dim)
    base = 1 - thresholds["quantile"]
    brier_ref = ((base - event) ** 2).where(mask).mean(dim)
    return xr.Dataset({"crps": crps, "crps_ref": crps_ref, "crpss": 1 - crps / crps_ref, "brier": brier, "bss": 1 - brier / brier_ref})
