"""
Does a method beat M1 (persistence of the reference period, the operational method)? `significance_m1.nc`.

The surrogate test (`surrogates.py`) asks whether a method has information beyond M0; its null keeps the
autocorrelation of the observations, so it cannot test "better than persistence" (the advantage of M0 over
M1 comes from that autocorrelation and is in its null). Here the question is whether the expected score
difference to M1 is positive, tested with a paired moving-block bootstrap over the init years:

  crpss    = 1 - sum_Y CRPS(m) / sum_Y CRPS(M1)          (the CRPSS of the score maps)
  msss_m1  = 1 - sum_Y (pred - obs)^2 / sum_Y obs^2        (mean-squared-error skill, M1 predicts 0)

on the samples both cover. The init years are resampled in blocks of max(n^(1/3), window length) (overlapping
windows), with the same draws for all grid points (spatial correlation kept for the field test) and methods.
One-sided p-value of H0 "no improvement" (statistic <= 0) from the centred bootstrap distribution:
p = (1 + #{S* - S >= S}) / (n + 1). FDR (Benjamini-Hochberg) at q = fdr_factor * alpha per map, field test
with the area-weighted (cos lat) mean of the statistic in the same way. No area-fraction test here.
"""

import logging

import numpy as np
import xarray as xr
from joblib import Parallel, delayed

from climate_scenarios_2050.core.timeconv import window_label
from climate_scenarios_2050.evaluate.surrogates import METHOD_ORDER, _crps, fdr_mask

log = logging.getLogger(__name__)

STATISTICS = ["crpss", "msss_m1"]


def block_counts(n: int, block: int, n_boot: int, rng) -> np.ndarray:
    """(n_boot, n) how often each sample is drawn by a moving-block bootstrap of length n"""
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None]).reshape(n_boot, -1)[:, :n]
    counts = np.zeros((n_boot, n))
    np.add.at(counts, (np.arange(n_boot)[:, None], idx), 1)
    return counts


def ratio_test(num_m, num_ref, weights, counts, area, chunk=8000):
    """
    statistic 1 - sum(w num_m) / sum(w num_ref) per point and its bootstrap: (observed (P,), local p (P,),
    field mean, field p); num_* (init, P), weights (init, P) in {0, 1}, counts (n_boot, init); points in chunks
    """
    a, b = num_m * weights, num_ref * weights
    with np.errstate(invalid="ignore", divide="ignore"):
        observed = 1 - a.sum(0) / b.sum(0)
    ok = np.isfinite(observed) & (weights.sum(0) >= 3)
    observed = np.where(ok, observed, np.nan)
    p = np.full(observed.shape, np.nan)
    w = area * ok
    fsum, fweight = np.zeros(len(counts)), np.zeros(len(counts))
    for s0 in range(0, observed.size, chunk):
        sl = slice(s0, s0 + chunk)
        with np.errstate(invalid="ignore", divide="ignore"):
            boot = 1 - (counts @ a[:, sl]) / (counts @ b[:, sl])
        fin = np.isfinite(boot)
        exceed = (((boot - observed[sl]) >= observed[sl]) & fin).sum(0)
        valid = fin.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            p[sl] = np.where(ok[sl] & (valid > 0), (1 + exceed) / (valid + 1), np.nan)
        fsum += (np.where(fin, boot, 0) * w[sl]).sum(1)
        fweight += (fin * w[sl]).sum(1)
    if w.sum() == 0:
        return observed, p, np.nan, np.nan
    field = np.nansum(observed * w) / w.sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        fb = fsum / fweight
    field_p = (1 + ((fb - field) >= field).sum()) / (np.isfinite(fb).sum() + 1)
    return observed, p, field, field_p


def _window_job(cases, kinds, names, block, n_boot, seed, area):
    """the tests of one window: arrays (statistic, kind, method, point) and (statistic, kind, method)"""
    n_points = len(area)
    counts = block_counts(len(cases[kinds[0]]["target"]), block, n_boot, np.random.default_rng(seed))
    val = np.full((len(STATISTICS), len(kinds), len(names), n_points), np.nan)
    pl = np.full_like(val, np.nan)
    fm, fp = np.full(val.shape[:-1], np.nan), np.full(val.shape[:-1], np.nan)
    for ik, kind in enumerate(kinds):
        t = cases[kind]["target"]
        crps1 = _crps(*cases[kind]["M1"], t)
        for im, name in enumerate(names):
            if name not in cases[kind]["methods"]:
                continue
            mean, spread = cases[kind]["methods"][name]
            w = (np.isfinite(mean) & np.isfinite(spread) & np.isfinite(t) & np.isfinite(crps1)).astype(float)
            t0, m0 = np.nan_to_num(t), np.nan_to_num(mean)
            crps = np.nan_to_num(_crps(m0, np.where(w > 0, spread, 1.0), t0))
            for js, (num_m, num_ref) in enumerate([(crps, np.nan_to_num(crps1)), ((m0 - t0) ** 2, t0 ** 2)]):
                val[js, ik, im], pl[js, ik, im], fm[js, ik, im], fp[js, ik, im] = ratio_test(num_m, num_ref, w, counts, area)
    return val, pl, fm, fp


def m1_test(cfg, targets, methods, windows) -> xr.Dataset:
    scfg, boot = cfg["evaluation"]["significance"], cfg["evaluation"]["bootstrap"]
    labels = [window_label(w) for w in windows]
    kinds = list(targets)
    names = sorted({m for kind in kinds for m in methods[kind] if m != "M1"}, key=METHOD_ORDER.index)
    template = targets[kinds[0]].transpose("init_year", "window", "latitude", "longitude")
    inits = template.init_year.values
    n_lat, n_lon = template.sizes["latitude"], template.sizes["longitude"]
    area = np.repeat(np.cos(np.deg2rad(template.latitude.values)), n_lon)
    rng = np.random.default_rng(boot["seed"])
    q = scfg["fdr_factor"] * scfg["alpha"]
    shape = (len(STATISTICS), len(kinds), len(names), len(labels))
    value, p_local = np.full(shape + (n_lat * n_lon,), np.nan), np.full(shape + (n_lat * n_lon,), np.nan)
    field_mean, p_field = np.full(shape, np.nan), np.full(shape, np.nan)

    def arr(da, label):
        return da.broadcast_like(template).transpose("init_year", "window", "latitude", "longitude").sel(window=label).values.reshape(len(inits), -1)

    # one job per window (own block length and draws); the arrays of a window are small
    seeds = rng.integers(0, 2**31, size=len(windows))
    jobs = []
    for iw, ((a, b), label) in enumerate(zip(windows, labels)):
        block = max(int(round(len(inits) ** (1 / 3))), b - a + 1)
        cases = {}
        for kind in kinds:
            cases[kind] = {"target": arr(targets[kind], label), "M1": tuple(arr(x, label) for x in methods[kind]["M1"]),
                           "methods": {name: tuple(arr(x, label) for x in methods[kind][name]) for name in names if name in methods[kind]}}
        jobs.append(delayed(_window_job)(cases, kinds, names, block, boot["n"], int(seeds[iw]), area))
    results = Parallel(n_jobs=cfg["resources"]["n_jobs"])(jobs)
    for iw, (val, pl, fm, fp) in enumerate(results):
        value[:, :, :, iw], p_local[:, :, :, iw], field_mean[:, :, :, iw], p_field[:, :, :, iw] = val, pl, fm, fp
    significant = np.zeros(p_local.shape, dtype=bool)
    for idx in np.ndindex(*p_local.shape[:-1]):
        significant[idx] = fdr_mask(p_local[idx], q)
    dims = ("statistic", "target", "fc_method", "window")
    sdims = dims + ("latitude", "longitude")
    full = shape + (n_lat, n_lon)
    ds = xr.Dataset(
        {
            "value": (sdims, value.reshape(full).astype("float32")),
            "p_local": (sdims, p_local.reshape(full).astype("float32")),
            "significant": (sdims, significant.reshape(full).astype("int8")),
            "field_mean": (dims, field_mean),
            "p_field_mean": (dims, p_field),
        },
        coords={"statistic": STATISTICS, "target": kinds, "fc_method": names, "window": labels,
                "latitude": template.latitude.values, "longitude": template.longitude.values},
    )
    ds.attrs.update({"null": "no improvement over M1 (expected score difference <= 0); paired moving-block bootstrap over the init years",
                     "n_bootstrap": boot["n"], "seed": boot["seed"], "alpha": scfg["alpha"], "fdr_q": q})
    return ds
