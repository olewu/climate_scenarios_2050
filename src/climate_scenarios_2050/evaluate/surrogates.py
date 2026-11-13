"""
Significance of the wind skill maps beyond the reference method M0, with spatially coherent
phase-randomized surrogates of the observed annual internal wind (`evaluation.significance`).

Null hypothesis: the predictions contain no information on the observations beyond M0
(reversion to the long-term mean). The model-based part of every prediction is kept fixed, the
observed internal annual wind u(t) is replaced by surrogates u*(t) with the same power spectrum
at every grid point and the same cross-spectra between grid points (one random phase rotation per
frequency shared by all points, Prichard & Theiler 1994). This keeps the autocorrelation of the
annual series, and through the overlapping windows and the shared 20-year reference that of the
evaluated samples, as well as the spatial correlation needed for the field test.

Everything that depends on the observations is rebuilt from d = u* - u. The observed internal
window and reference means come from a forced fit that leaves out the target window and the
reference period (`split.obs_leakage: leave_target_out`). This fit is linear in the data, so for
every (init, window) the internal window mean and reference mean are fixed linear operators
(window operator minus the forced part fitted without those years) applied to the annual series;
applying them to d repeats the leave-out split for every surrogate (excluded years keep their
anomaly, which raises the correlation of reference and window means and has to be in the null too):
  target*  = target + (WL d - RL d) * to_units
  pred_m*  = pred_m - c_m * RL d * to_units, c_m = 1 for the methods subtracting the observed
             internal reference (M0, M2b, M3b, *_b, perfect_nao), 0 for M1, M2a, M3a
  M1 spread recomputed from target* (also the CRPSS reference)
With d = 0 this reproduces the actual scores. The regression parameters of the linear model are
kept fixed (the test is conditional on the predictions), as is the conversion to % (to_units).

Statistics per (target, method, window, grid point), all on the samples both method and M0 cover:
  dcrpss     (CRPS_M0 - CRPS_m) / CRPS_M1                (= CRPSS_m - CRPSS_M0, primary)
  dcorr      corr(pred_m, target) - corr(pred_M0, target)
  corr_anom  corr(pred_m - pred_M0, target - pred_M0)    (information beyond M0; for M2b/M3b on
             the internal target this is corr(P, W) of the predicted and observed window levels)
One-sided local p-values (1 + #{null >= observed}) / (n + 1); grid points significant after the
false discovery rate (Benjamini-Hochberg) at q = fdr_factor * alpha (Wilks 2016). Field tests per map
with the area-weighted (cos lat) mean of the statistic and the area fraction where it is positive.
"""

import logging

import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from scipy.special import ndtr

from climate_scenarios_2050.core.timeconv import window_label
from climate_scenarios_2050.evaluate import metrics
from climate_scenarios_2050.preprocess import obs
from climate_scenarios_2050.split.run import SplitResults

log = logging.getLogger(__name__)

STATISTICS = ["dcrpss", "dcorr", "corr_anom"]
# methods whose prediction contains "- observed internal reference"
WITH_REFERENCE = {"M0", "M2b", "M3b", "M2b_b", "M3b_b", "perfect_nao"}


#---------------surrogates--------------#
def coherent_phases(n_years: int, n: int, seed: int) -> np.ndarray:
    """random phase rotations (n, n_freq), shared by all grid points; mean (and Nyquist) unchanged"""
    n_freq = n_years // 2 + 1
    phases = np.random.default_rng(seed).uniform(0, 2 * np.pi, size=(n, n_freq))
    phases[:, 0] = 0
    if n_years % 2 == 0:
        phases[:, -1] = 0
    return phases


def perturbation(spectrum: np.ndarray, phases: np.ndarray, n_years: int) -> np.ndarray:
    """d = u* - u (years, points) for the spectrum of u (freq, points) and one phase rotation"""
    return np.fft.irfft(spectrum * (np.exp(1j * phases)[:, None] - 1), n=n_years, axis=0)


def averaging_operators(years, init_years, windows, reference_length):
    """window and reference averaging matrices (init, window, year), lag 0"""
    years = np.asarray(years)
    win = np.zeros((len(init_years), len(windows), len(years)))
    ref = np.zeros((len(init_years), len(windows), len(years)))
    for ii, init in enumerate(init_years):
        in_ref = (years > init - reference_length) & (years <= init)
        ref[ii] = in_ref / max(in_ref.sum(), 1)
        for iw, (a, b) in enumerate(windows):
            in_win = (years >= init + a) & (years <= init + b)
            win[ii, iw] = in_win / max(in_win.sum(), 1)
    return win, ref


def leave_out_operators(win, ref, design: np.ndarray, excluded: np.ndarray, min_fit_years: int):
    """
    operators giving the window/reference means of the internal series, with the forced part
    fitted without the excluded years of each case: M - (M X) pinv(X_keep) S_keep.
    design (year, term), excluded (init, window, year) or (year,) for all cases
    """
    valid = np.isfinite(design).all(axis=1)
    X = np.where(valid[:, None], design, 0.0)
    win_lo, ref_lo = np.full(win.shape, np.nan), np.full(ref.shape, np.nan)
    cache = {}
    for ii, iw in np.ndindex(*win.shape[:2]):
        keep = valid & ~(excluded if excluded.ndim == 1 else excluded[ii, iw])
        if keep.sum() < min_fit_years:
            continue
        key = keep.tobytes()
        if key not in cache:
            fit = np.zeros((X.shape[1], len(keep)))
            fit[:, keep] = np.linalg.pinv(X[keep])
            cache[key] = X @ fit   # (year, year): fitted forced part
        hat = cache[key]
        win_lo[ii, iw] = win[ii, iw] - win[ii, iw] @ hat
        ref_lo[ii, iw] = ref[ii, iw] - ref[ii, iw] @ hat
    return win_lo, ref_lo


def observation_operators(cfg, scenario, years, init_years, windows):
    """operators for the observed internal window and reference means as used in the evaluation"""
    ref_len = cfg["target"]["reference_length"]
    win, ref = averaging_operators(years, init_years, windows, ref_len)
    design = SplitResults(cfg, scenario).wind_design(years).values
    if cfg["split"]["obs_leakage"] == "full_period":
        excluded = np.zeros(len(years), dtype=bool)
    else:
        from climate_scenarios_2050.split import forced as fc

        excluded = fc.window_exclusions(years, init_years, windows, lag=0, reference_length=ref_len).values
    win_lo, ref_lo = leave_out_operators(win, ref, design, excluded, cfg["split"]["min_fit_years"])
    # cases without a fit have no actual scores either (masked); zeros keep NaN out of the sums
    return np.nan_to_num(win_lo), np.nan_to_num(ref_lo)


def observed_internal_wind(cfg, scenario, wind_variable):
    """observed annual internal wind, forced part fitted to all years (year, latitude, longitude)"""
    split = SplitResults(cfg, scenario)
    wind = obs.open_obs_wind(cfg, wind_variable)
    coef = split.wind_coef(wind_variable, "coef_full").load()
    internal = wind - (split.wind_design(wind.year.values) * coef).sum("term", skipna=False)
    return internal.dropna("year", how="all")


#---------------statistics--------------#
def _corr_from_moments(n, sx, sy, sxx, syy, sxy):
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = sxy - sx * sy / n
        return cov / np.sqrt((sxx - sx**2 / n) * (syy - sy**2 / n))


def correlations(pred, p0, target, w):
    """corr(pred, t), corr(p0, t) and corr(pred - p0, t - p0) along the init axis (2), weights w in {0, 1}"""
    a, b, t = pred * w, p0 * w, target * w   # w is 0/1, so a * pred = w * pred**2
    n = w.sum(axis=2)
    sa, sb, st = a.sum(axis=2), b.sum(axis=2), t.sum(axis=2)
    saa, sbb, stt = (a * pred).sum(axis=2), (b * p0).sum(axis=2), (t * target).sum(axis=2)
    sab, sat, sbt = (a * p0).sum(axis=2), (a * target).sum(axis=2), (b * target).sum(axis=2)
    c_m = _corr_from_moments(n, sa, st, saa, stt, sat)
    c_0 = _corr_from_moments(n, sb, st, sbb, stt, sbt)
    anom = _corr_from_moments(n, sa - sb, st - sb, saa - 2 * sab + sbb, stt - 2 * sbt + sbb, sat - sab - sbt + sbb)
    return c_m, c_0, anom


def _crps(mu, sigma, obs_):
    with np.errstate(invalid="ignore", divide="ignore"):
        z = (obs_ - mu) / sigma
        return sigma * (z * (2 * ndtr(z) - 1) + 2 * np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi) - 1 / np.sqrt(np.pi))


class Problem:
    """
    actual targets and predictions of one point chunk as arrays
    target (kind, init, window, point), pred/spread (kind, method, init, window, point)
    """

    def __init__(self, target, pred, spread, joint, methods, to_units, separated):
        f = np.float32
        self.target = np.nan_to_num(target).astype(f)
        self.pred = np.nan_to_num(pred).astype(f)
        self.spread = np.where(np.isfinite(spread), spread, 1.0).astype(f)
        self.joint = joint.astype(f)              # samples covered by method, M0 and the target
        self.valid_target = np.isfinite(target).astype(f)
        self.methods = methods
        self.m0 = methods.index("M0")
        self.m1 = methods.index("M1")
        self.c = np.array([1.0 if m in WITH_REFERENCE else 0.0 for m in methods], dtype=f)
        self.to_units = np.nan_to_num(to_units).astype(f)   # (init, point)
        self.separated = separated.astype(f)                # (window, init, init)

    def statistics(self, d_window=None, d_reference=None) -> np.ndarray:
        """(statistic, kind, method, window, point) for a perturbation (window/reference means of d)"""
        target, pred = self.target, self.pred
        if d_window is not None:
            # (init, window, point)
            d_window, d_reference = d_window.astype(np.float32), d_reference.astype(np.float32)
            target = target + ((d_window - d_reference) * self.to_units[:, None])[None]
            pred = pred - self.c[None, :, None, None, None] * (d_reference * self.to_units[:, None])[None, None]
        # M1: spread of the observed changes around zero, non-overlapping samples
        sq = np.einsum("wij,kjwp->kiwp", self.separated, target**2 * self.valid_target)
        cnt = np.einsum("wij,kjwp->kiwp", self.separated, self.valid_target)
        with np.errstate(invalid="ignore", divide="ignore"):
            spread = self.spread.copy()
            spread[:, self.m1] = np.sqrt(sq / cnt)
        crps = _crps(pred, spread, target[:, None])
        w = self.joint
        with np.errstate(invalid="ignore", divide="ignore"):
            num = (crps * w).sum(axis=2)
            num0 = (crps[:, self.m0:self.m0 + 1] * w).sum(axis=2)
            num1 = (crps[:, self.m1:self.m1 + 1] * w).sum(axis=2)
            dcrpss = (num0 - num) / num1
        c_m, c_0, anom = correlations(pred, pred[:, self.m0:self.m0 + 1], target[:, None], w)
        out = np.stack([dcrpss, c_m - c_0, anom])
        out[:, :, self.m0] = np.nan
        out[:, w.sum(axis=2) < 3] = np.nan
        return out


def _chunk(problem: Problem, spectrum, phases, n_years, win_op, ref_op, area):
    """observed statistics, exceedance counts and per-surrogate partial field sums of one chunk"""
    observed = problem.statistics()
    finite = np.isfinite(observed)
    exceed = np.zeros(observed.shape, dtype="int32")
    n_null = np.zeros(observed.shape, dtype="int32")
    n = len(phases)
    field = {key: np.zeros((n, *observed.shape[:-1])) for key in ["sum", "weight", "positive"]}
    assert np.isfinite(win_op).all() and np.isfinite(ref_op).all()
    for b in range(n):
        d = perturbation(spectrum, phases[b], n_years)
        null = problem.statistics(np.einsum("iwy,yp->iwp", win_op, d), np.einsum("iwy,yp->iwp", ref_op, d))
        ok = np.isfinite(null)
        exceed += (ok & finite & (null >= observed)).astype("int32")
        n_null += ok
        a = area * ok
        field["sum"][b] = (np.where(ok, null, 0) * area).sum(axis=-1)
        field["weight"][b] = a.sum(axis=-1)
        field["positive"][b] = (a * (np.where(ok, null, 0) > 0)).sum(axis=-1)
    obs_field = {
        "sum": (np.where(finite, observed, 0) * area).sum(axis=-1),
        "weight": (area * finite).sum(axis=-1),
        "positive": (area * finite * (np.where(finite, observed, 0) > 0)).sum(axis=-1),
    }
    return observed, exceed, n_null, field, obs_field


def _quantile(x, q):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanquantile(x, q, axis=0)


def fdr_mask(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini-Hochberg over the finite p-values of one map"""
    flat = p.ravel()
    ok = np.isfinite(flat)
    sig = np.zeros(flat.shape, dtype=bool)
    m = ok.sum()
    if m == 0:
        return sig.reshape(p.shape)
    order = np.argsort(flat[ok])
    below = np.nonzero(flat[ok][order] <= q * np.arange(1, m + 1) / m)[0]
    if len(below):
        idx = np.nonzero(ok)[0][order[: below[-1] + 1]]
        sig[idx] = True
    return sig.reshape(p.shape)


def surrogate_test(cfg, scenario, wind_variable, targets, methods, to_units) -> xr.Dataset:
    scfg = cfg["evaluation"]["significance"]
    windows = [tuple(w) for w in cfg["target"]["windows"]]
    labels = [window_label(w) for w in windows]
    kinds = list(targets)
    names = sorted({m for kind in kinds for m in methods[kind]}, key=lambda m: ["M0", "M1", "M2a", "M3a", "M2b", "M3b", "M2b_b", "M3b_b", "perfect_nao"].index(m))

    template = targets[kinds[0]].transpose("init_year", "window", "latitude", "longitude")
    inits = template.init_year.values
    space = template.isel(init_year=0, window=0, drop=True)
    n_lat, n_lon = space.shape

    def arr(da):
        return da.transpose("init_year", "window", "latitude", "longitude").sel(window=labels).values.reshape(len(inits), len(labels), -1)

    nan = np.full((len(inits), len(labels), n_lat * n_lon), np.nan)
    target = np.stack([arr(targets[k]) for k in kinds])
    pred = np.stack([np.stack([arr(methods[k][m][0].broadcast_like(template)) if m in methods[k] else nan for m in names]) for k in kinds])
    spread = np.stack([np.stack([arr(methods[k][m][1].broadcast_like(template)) if m in methods[k] else nan for m in names]) for k in kinds])
    m0 = names.index("M0")
    joint = np.isfinite(pred) & np.isfinite(pred[:, m0:m0 + 1]) & np.isfinite(target[:, None]) & np.isfinite(spread)
    tu = to_units.sel(init_year=inits).transpose("init_year", "latitude", "longitude").values.reshape(len(inits), -1)

    # observed internal wind, contiguous years covering all windows and references
    u = observed_internal_wind(cfg, scenario, wind_variable).transpose("year", "latitude", "longitude")
    years = u.year.values
    if np.any(np.diff(years) != 1):
        raise ValueError(f"observed internal wind has gaps: {years}")
    u = u.values.reshape(len(years), -1)
    with np.errstate(invalid="ignore"), __import__("warnings").catch_warnings():
        __import__("warnings").simplefilter("ignore", RuntimeWarning)
        u = u - np.nanmean(u, axis=0)
    spectrum = np.fft.rfft(np.nan_to_num(u), axis=0)
    win_op, ref_op = observation_operators(cfg, scenario, years, inits, windows)
    separated = np.stack([metrics.separated(inits, b - a + 1) for a, b in windows]).astype(float)
    phases = coherent_phases(len(years), scfg["n"], scfg["seed"])
    area = np.repeat(np.cos(np.deg2rad(space.latitude.values)), n_lon)

    chunk = scfg["chunk_points"]
    starts = range(0, n_lat * n_lon, chunk)
    log.info(f"surrogate test {scenario} {wind_variable}: {scfg['n']} surrogates, {len(names)} methods, {n_lat * n_lon} points in {len(starts)} chunks")
    jobs = []
    for s in starts:
        sl = slice(s, s + chunk)
        problem = Problem(target[..., sl], pred[..., sl], spread[..., sl], joint[..., sl], names, tu[:, sl], separated)
        jobs.append(delayed(_chunk)(problem, spectrum[:, sl], phases, len(years), win_op, ref_op, area[sl]))
    results = Parallel(n_jobs=cfg["resources"]["n_jobs"], verbose=5)(jobs)

    observed = np.concatenate([r[0] for r in results], axis=-1)
    exceed = np.concatenate([r[1] for r in results], axis=-1)
    n_null = np.concatenate([r[2] for r in results], axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p_local = np.where(np.isfinite(observed) & (n_null > 0), (1 + exceed) / (n_null + 1), np.nan)
    q = scfg["fdr_factor"] * scfg["alpha"]
    significant = np.zeros(p_local.shape, dtype=bool)
    for idx in np.ndindex(*p_local.shape[:-1]):
        significant[idx] = fdr_mask(p_local[idx], q)

    field = {key: sum(r[3][key] for r in results) for key in ["sum", "weight", "positive"]}
    obs_field = {key: sum(r[4][key] for r in results) for key in ["sum", "weight", "positive"]}
    with np.errstate(invalid="ignore", divide="ignore"):
        null_mean, null_frac = field["sum"] / field["weight"], field["positive"] / field["weight"]
        obs_mean, obs_frac = obs_field["sum"] / obs_field["weight"], obs_field["positive"] / obs_field["weight"]

    def field_p(observed_, null):
        valid = np.isfinite(null).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            p = (1 + (null >= observed_[None]).sum(axis=0)) / (valid + 1)
        return np.where(np.isfinite(observed_) & (valid > 0), p, np.nan)

    dims = ("statistic", "target", "fc_method", "window")
    coords = {"statistic": STATISTICS, "target": kinds, "fc_method": names, "window": labels}
    shape = observed.shape[:-1] + (n_lat, n_lon)
    sdims = dims + ("latitude", "longitude")
    ds = xr.Dataset(
        {
            "value": (sdims, observed.reshape(shape).astype("float32")),
            "p_local": (sdims, p_local.reshape(shape).astype("float32")),
            "significant": (sdims, significant.reshape(shape).astype("int8")),
            "field_mean": (dims, obs_mean),
            "p_field_mean": (dims, field_p(obs_mean, null_mean)),
            "area_fraction_positive": (dims, obs_frac),
            "p_area_fraction": (dims, field_p(obs_frac, null_frac)),
            "null_field_mean_q95": (dims, _quantile(null_mean, 0.95)),
        },
        coords={**coords, "latitude": space.latitude.values, "longitude": space.longitude.values},
    )
    ds.attrs.update({
        "null": "no information beyond M0; spatially coherent phase-randomized surrogates of the observed annual internal wind",
        "n_surrogates": scfg["n"], "seed": scfg["seed"], "alpha": scfg["alpha"], "fdr_q": q,
        "surrogate_years": f"{years[0]}-{years[-1]}",
    })
    return ds
