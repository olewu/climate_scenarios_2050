"""
Compute the predictor index for all observational datasets and all CMIP6 members/scenarios,
and the merged observational index.

Output (`<root>/indices/<hash>/`):
- `pattern_<source>.nc`            reference pattern (EOF method)
- `patterns_own/<dataset or model>_<scenario>.nc`   own patterns (pattern_source: own)
- `obs_<dataset>.nc`               index (year) per observational dataset
- `obs_merged.nc`                  merged index + frozen harmonization parameters
- `cmip6_<scenario>.nc`            index (member, year) with model coordinates
"""

import logging

import numpy as np
import xarray as xr

from climate_scenarios_2050.core.io import StepOutput, write_netcdf
from climate_scenarios_2050.preprocess import cmip6, obs
from climate_scenarios_2050.preprocess.indices import box, eof

log = logging.getLogger(__name__)

IMPLEMENTED = ["eof", "box"]


def index_output(cfg) -> StepOutput:
    pred = cfg["predictor"]
    cmip6_store, _ = cmip6.field_store(cfg, pred["variable"], pred["aggregation"])
    return StepOutput(
        cfg, "indices", sections=["predictor"],
        extra={"obs_datasets": cfg["preprocess"]["obs_datasets"], "merge": cfg["obs"]["merge"]},
        upstream={"obs_fields": obs.obs_field_output(cfg, pred["variable"]).hash, "cmip6_fields": cmip6_store.name},
    )


def _domain(field, domain):
    return field.sel(lat=slice(*domain["lat"]), lon=slice(*domain["lon"]))


class IndexCalculator:
    """computes the configured index for any field, keeping the reference pattern/statistics"""

    def __init__(self, cfg, obs_fields: dict, out: StepOutput):
        self.cfg = cfg
        self.pred = cfg["predictor"]
        self.out = out
        if self.pred["method"] not in IMPLEMENTED:
            raise NotImplementedError(f"index method {self.pred['method']} is not implemented yet")
        self.ref_name = "ERA5" if self.pred["pattern_source"] == "own" else self.pred["pattern_source"]
        ref_field = _domain(obs_fields[self.ref_name], self.pred["domain"])
        self.ref_clim = eof.normal_period_climatology(ref_field, self.pred["normal_period"])
        if self.pred["method"] == "eof":
            self.reference = eof.fit_pattern(
                ref_field, self.pred["normal_period"], self.pred["n_modes"], self.pred["detrend_for_fit"],
                self.pred["sign_reference_point"], source=self.ref_name,
            )
            fl = out.path(f"pattern_{self.ref_name}.nc")
            if not fl.exists():
                write_netcdf(self.reference, fl, cfg)
        else:
            self.reference = box.reference_stats(ref_field, self.pred["box"], self.pred["normal_period"], self.ref_clim)

    def __call__(self, field: xr.DataArray, name: str) -> xr.DataArray:
        field = _domain(field, self.pred["domain"])
        clim = eof.normal_period_climatology(field, self.pred["normal_period"])
        if self.pred["method"] == "box":
            return box.project(field, self.reference, self.pred["box"], clim)
        mode = self.pred["mode"]
        pattern = self.reference
        if self.pred["pattern_source"] == "own":
            own = eof.fit_pattern(field, self.pred["normal_period"], self.pred["n_modes"], self.pred["detrend_for_fit"], source=name)
            pattern = eof.match_modes(own, self.reference, [mode])
            fl = self.out.path(f"patterns_own/{name}.nc")
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
    obs_fields = {ds: obs.open_obs_field(cfg, ds, pred["variable"], pred["aggregation"]) for ds in cfg["preprocess"]["obs_datasets"]}
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
    if not fl.exists():
        merged = merge_indices(obs_indices[merge_cfg["datasets"][0]], obs_indices[merge_cfg["datasets"][1]], merge_cfg)
        log.info(f"merged index: overlap corr {merged.attrs['overlap_corr']:.3f}, rmse {merged.attrs['overlap_rmse']:.3f}")
        write_netcdf(merged, fl, cfg)

    for scenario in scenarios or cfg["cmip6"]["scenarios"]:
        fl = out.path(f"cmip6_{scenario}.nc")
        if fl.exists():
            log.info(f"{fl} exists, skipping")
            continue
        try:
            field = cmip6.open_field(cfg, pred["variable"], pred["aggregation"], scenario)[pred["variable"]]
        except FileNotFoundError as err:
            log.warning(str(err))
            continue
        indices = []
        for model in np.unique(field["model"].values):
            sub = field.isel(member=(field["model"] == model).values).load()
            indices.append(calc(sub, f"{model}_{scenario}").compute())
        idx = xr.concat(indices, dim="member")
        write_netcdf(idx.to_dataset(), fl, cfg)
        log.info(f"{scenario}: index for {idx.sizes['member']} members")
    return out


def open_obs_index(cfg, dataset: str) -> xr.DataArray:
    out = index_output(cfg)
    with xr.open_dataset(out.path(f"obs_{dataset}.nc")) as ds:
        return ds["index"].load()


def open_cmip6_index(cfg, scenario: str) -> xr.DataArray:
    out = index_output(cfg)
    with xr.open_dataset(out.path(f"cmip6_{scenario}.nc")) as ds:
        return ds["index"].load()
