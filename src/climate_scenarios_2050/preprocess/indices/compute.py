"""
Compute the predictor index for all observational datasets and all CMIP6 members/scenarios,
and the merged observational index.

With `predictor.remove_global_mean` the global-mean field (`<variable>_global_mean`, e.g. global-mean
SST for the PDO) is subtracted from every field before fitting and projecting the pattern. With
`predictor.mask: ocean` the field is restricted to the ocean (mask from ERA5 SST), e.g. for a PDO-like
index from near-surface air temperature.

Output of the observations (`<root>/indices/<hash>/`, independent of the CMIP6 archive):
- `pattern_<source>.nc`            reference pattern (EOF method)
- `patterns_own/<dataset>.nc`      own patterns (pattern_source: own)
- `obs_<dataset>.nc`               index (year) per observational dataset
- `obs_merged.nc`                  merged index + frozen harmonization parameters (if `obs.merge` is set)

Output per scenario (`<root>/indices_cmip6/<hash>/`, hash includes the scenario's member list, so a
download for one scenario leaves the others' hashes unchanged):
- `cmip6_<scenario>.nc`            index (member, year) with model coordinates
- `patterns_own/<model>_<scenario>.nc`
"""

import logging

import numpy as np
import xarray as xr

from climate_scenarios_2050.core import domain as dom
from climate_scenarios_2050.core.config import predictor_field
from climate_scenarios_2050.core.io import StepOutput, write_netcdf
from climate_scenarios_2050.preprocess import cmip6, masks, obs
from climate_scenarios_2050.preprocess.indices import box, eof

log = logging.getLogger(__name__)

IMPLEMENTED = ["eof", "box"]
PLOT_ONLY = {"predictor": ["reference_skill"]}   # settings that only change figures


def index_output(cfg) -> StepOutput:
    """observed indices and reference pattern"""
    pred = cfg["predictor"]
    return StepOutput(
        cfg, "indices", sections=["predictor"],
        extra={"merge": cfg["obs"]["merge"]},
        upstream={"obs_fields": obs.obs_field_output(cfg, predictor_field(cfg)).hash},
        exclude=PLOT_ONLY,
    )


def cmip6_index_output(cfg, scenario: str) -> StepOutput:
    """CMIP6 indices of one scenario"""
    pred = cfg["predictor"]
    return StepOutput(
        cfg, "indices_cmip6", sections=["predictor"], extra={"scenario": scenario},
        upstream={"indices": index_output(cfg).hash, "cmip6_fields": cmip6.field_id(cfg, predictor_field(cfg), pred["aggregation"], scenario)},
        exclude=PLOT_ONLY,
    )


def _domain(field, domain):
    return dom.select(field, domain)


def remove_global_mean(cfg, field: xr.DataArray, gm) -> xr.DataArray:
    """field minus its global mean if `predictor.remove_global_mean` (gm: None or (member,) year)"""
    if not cfg["predictor"].get("remove_global_mean"):
        return field
    if gm is None:
        raise ValueError(f"predictor.remove_global_mean: no global mean stored with the {cfg['predictor']['variable']} field")
    return field - gm


class IndexCalculator:
    """computes the configured index for any field, keeping the reference pattern/statistics"""

    def __init__(self, cfg, obs_fields: dict, out: StepOutput):
        self.cfg = cfg
        self.pred = cfg["predictor"]
        self.out = out
        if self.pred["method"] not in IMPLEMENTED:
            raise NotImplementedError(f"index method {self.pred['method']} is not implemented yet")
        self.ref_name = "ERA5" if self.pred["pattern_source"] == "own" else self.pred["pattern_source"]
        ref_field = masks.apply(cfg, _domain(obs_fields[self.ref_name], self.pred["domain"]), self.pred.get("mask"))
        self.ref_clim = eof.normal_period_climatology(ref_field, self.pred["normal_period"])
        if self.pred["method"] == "eof":
            sign_point = self.pred["sign_reference_point"]
            if sign_point is not None:
                sign_point = {"lat": sign_point["lat"], "lon": dom.point_lon(sign_point["lon"], self.pred["domain"])}
            self.reference = eof.fit_pattern(
                ref_field, self.pred["normal_period"], self.pred["n_modes"], self.pred["detrend_for_fit"],
                sign_point, source=self.ref_name,
            )
            fl = out.path(f"pattern_{self.ref_name}.nc")
            if not fl.exists():
                write_netcdf(self.reference, fl, cfg)
        else:
            self.reference = box.reference_stats(ref_field, self.pred["box"], self.pred["normal_period"], self.ref_clim)

    def __call__(self, field: xr.DataArray, name: str, out: StepOutput = None) -> xr.DataArray:
        """`out`: where own patterns are written (default: the observed-index output)"""
        field = masks.apply(self.cfg, _domain(field, self.pred["domain"]), self.pred.get("mask"))
        clim = eof.normal_period_climatology(field, self.pred["normal_period"])
        if self.pred["method"] == "box":
            return box.project(field, self.reference, self.pred["box"], clim)
        mode = self.pred["mode"]
        pattern = self.reference
        if self.pred["pattern_source"] == "own":
            own = eof.fit_pattern(field, self.pred["normal_period"], self.pred["n_modes"], self.pred["detrend_for_fit"], source=name)
            pattern = eof.match_modes(own, self.reference, [mode])
            fl = (out or self.out).path(f"patterns_own/{name}.nc")
            if not fl.exists():
                write_netcdf(pattern, fl, self.cfg)
            log.info(f"{name}: own mode {int(pattern['own_mode'].sel(mode=mode))} matched (pattern corr {float(pattern['pattern_corr'].sel(mode=mode)):.2f}, fit years {pattern.attrs['fit_years']})")
        return eof.project(field, pattern, mode, clim)


def merge_indices(early: xr.DataArray, late: xr.DataArray, merge_cfg: dict) -> xr.Dataset:
    """
    rescale `late` to the mean/std of `early` over the harmonization period (frozen parameters,
    stored with the output) and merge: early before `year`, late from `year` on
    """
    period = slice(*merge_cfg["harmonization_period"])
    e, l = early.sel(year=period), late.sel(year=period)
    e, l = xr.align(e.dropna("year"), l.dropna("year"))
    params = {"early_mean": float(e.mean()), "early_std": float(e.std()), "late_mean": float(l.mean()), "late_std": float(l.std())}
    late_harm = (late - params["late_mean"]) * params["early_std"] / params["late_std"] + params["early_mean"]
    merged = xr.concat([early.sel(year=slice(None, merge_cfg["year"] - 1)), late_harm.sel(year=slice(merge_cfg["year"], None))], dim="year")
    source = xr.DataArray(np.where(merged.year < merge_cfg["year"], merge_cfg["datasets"][0], merge_cfg["datasets"][1]), dims="year", coords={"year": merged.year})
    ds = xr.Dataset({"index": merged, "source": source})
    ds.attrs = {**params, "merge_year": merge_cfg["year"], "harmonization_period": str(merge_cfg["harmonization_period"]),
                "overlap_corr": float(xr.corr(e, l)), "overlap_rmse": float(((e - l) ** 2).mean() ** .5)}
    return ds


def run(cfg, scenarios=None):
    pred = cfg["predictor"]
    out = index_output(cfg).prepare()
    log.info(f"indices -> {out.dir}")
    with_gm = bool(pred.get("remove_global_mean"))
    obs_fields = {}
    for ds in pred["obs_datasets"]:
        loaded = obs.open_obs_field(cfg, ds, predictor_field(cfg), pred["aggregation"], with_global_mean=with_gm)
        obs_fields[ds] = remove_global_mean(cfg, *loaded) if with_gm else loaded
    calc = IndexCalculator(cfg, obs_fields, out)

    obs_indices = {}
    for name, field in obs_fields.items():
        idx = calc(field, name).compute()
        obs_indices[name] = idx
        fl = out.path(f"obs_{name}.nc")
        if not fl.exists():
            write_netcdf(idx.to_dataset(), fl, cfg)

    merge_cfg = cfg["obs"]["merge"]
    fl = out.path("obs_merged.nc")
    if merge_cfg and not fl.exists():
        merged = merge_indices(obs_indices[merge_cfg["datasets"][0]], obs_indices[merge_cfg["datasets"][1]], merge_cfg)
        log.info(f"merged index: overlap corr {merged.attrs['overlap_corr']:.3f}, rmse {merged.attrs['overlap_rmse']:.3f}")
        write_netcdf(merged, fl, cfg)

    for scenario in scenarios or cfg["cmip6"]["scenarios"]:
        scen_out = cmip6_index_output(cfg, scenario).prepare()
        log.info(f"{scenario} indices -> {scen_out.dir}")
        fl = scen_out.path(f"cmip6_{scenario}.nc")
        if fl.exists():
            log.info(f"{fl} exists, skipping")
            continue
        try:
            data = cmip6.open_field(cfg, predictor_field(cfg), pred["aggregation"], scenario)
            field = remove_global_mean(cfg, data[pred["variable"]], data.get(f"{pred['variable']}_global_mean"))
        except FileNotFoundError as err:
            log.warning(str(err))
            continue
        indices = []
        for model in np.unique(field["model"].values):
            sub = field.isel(member=(field["model"] == model).values).load()
            indices.append(calc(sub, f"{model}_{scenario}", scen_out).compute())
        idx = xr.concat(indices, dim="member")
        write_netcdf(idx.to_dataset(), fl, cfg)
        log.info(f"{scenario}: index for {idx.sizes['member']} members")
    return out


def obs_index_names(cfg) -> list:
    """observed indices: one per dataset, plus the merged one if `obs.merge` is set"""
    return [*cfg["predictor"]["obs_datasets"], *(["merged"] if cfg["obs"].get("merge") else [])]


def open_obs_index(cfg, dataset: str) -> xr.DataArray:
    out = index_output(cfg)
    with xr.open_dataset(out.path(f"obs_{dataset}.nc")) as ds:
        return ds["index"].load()


def open_cmip6_index(cfg, scenario: str) -> xr.DataArray:
    out = cmip6_index_output(cfg, scenario)
    with xr.open_dataset(out.path(f"cmip6_{scenario}.nc")) as ds:
        return ds["index"].load()
