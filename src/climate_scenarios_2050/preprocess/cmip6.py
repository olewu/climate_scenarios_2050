"""
CMIP6 preprocessing: original monthly files -> seasonal/annual means on a common grid.

For every model and member with both a historical and a scenario run:
1. historical (until Dec 2014) and scenario (from Jan 2015) monthly data are joined, so seasons
   crossing the boundary (DJFM 2014) are complete
2. only members whose file names cover every month of `cmip6.full_period` are used (late historical
   starts such as EC-Earth3 r101+ from 1970 and short scenario runs are left out); for the variables of
   `cmip6.common_members` only the members complete in all of them, so all fields share one member list.
   Members with gaps in the data read are dropped as well
3. seasonal/annual means are computed on the native grid (month lengths of the model's own
   calendar, only complete seasons), restricted to the target domain plus a margin
4. the means are regridded (xesmf, weights cached per source grid) to the regular global grid,
   cut to the target domain (domains crossing the date line on longitudes 0 ... 360, see `core.domain`)
5. optionally (`global_mean: true`, or `ocean` for the ocean points only, mask from ERA5 SST) the
   area-weighted (cos lat) global mean of the native field is stored as `<variable>_global_mean`
   (member, year), e.g. to remove it from SST before the PDO EOF

Fields are entries of `preprocess.fields`, named after their CMIP6 variable unless `variable` is given
(e.g. `tas_np: {variable: tas, ...}`: North Pacific tas next to the regional tas of a target).

The CMIP6 table of a variable is `Amon` unless set in `cmip6.tables` (e.g. tos: Omon).

Output: one zarr per model, `<cmip6_processed>/<variable>_<aggregation>/<hash>/<scenario>/<model>.zarr`
with dims (member, year, lat, lon); member names `<model id><variant>` (ids from configs/model_ids.yaml).
"""

import functools
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import xesmf as xe
from joblib import Parallel, delayed

from climate_scenarios_2050.core import domain as dom
from climate_scenarios_2050.core.config import cmip6_table, config_hash, field_variable, load_model_ids, used_fields
from climate_scenarios_2050.core.io import ensure_writable, write_yaml
from climate_scenarios_2050.core.timeconv import seasonal_mean

log = logging.getLogger(__name__)

VARIANT_PATTERN = re.compile(r"_(r\d+i\d+p\d+f\d+)_")
HIST_END = 2014
DOMAIN_MARGIN = 5.0  # degrees added around the target domain before regridding
GRID_TOLERANCE = 0.01  # degrees, largest accepted coordinate difference between files of a member
ZERO_ALLOWED = {"pr", "tos"}  # variables where exact zeros are physical (otherwise a sign of unmapped cells)


#---------------grids--------------#
def target_grid(spacing: float, domain: dict = None) -> xr.Dataset:
    """
    regular global grid with cell centres at +-spacing/2 (lon -180 ... 180), optionally cut to a domain
    (a domain crossing the date line gets longitudes 0 ... 360)
    """
    lat = np.arange(-90 + spacing / 2, 90, spacing)
    lon = np.arange(-180 + spacing / 2, 180, spacing)
    grid = xr.Dataset(coords={"lat": ("lat", lat), "lon": ("lon", lon)})
    if domain is not None:
        grid = dom.select(grid, domain)
    grid["lat"].attrs = {"units": "degrees_north", "standard_name": "latitude"}
    grid["lon"].attrs = {"units": "degrees_east", "standard_name": "longitude"}
    return grid


def add_cell_bounds(grid: xr.Dataset) -> xr.Dataset:
    """lat_b/lon_b for conservative regridding of a regular grid"""
    out = grid.copy()
    for dim in ["lat", "lon"]:
        centres = grid[dim].values
        edges = np.concatenate([[centres[0] - (centres[1] - centres[0]) / 2], (centres[:-1] + centres[1:]) / 2, [centres[-1] + (centres[-1] - centres[-2]) / 2]])
        if dim == "lat":
            edges = np.clip(edges, -90, 90)
        out = out.assign_coords({f"{dim}_b": (f"{dim}_b", edges)})
    return out


def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """rename horizontal coordinates to lat/lon, longitudes to -180 ... 180, ascending order (1D grids)"""
    rename = {}
    for name in ds.coords:
        low = name.lower()
        if low in ["latitude", "nav_lat"]:
            rename[name] = "lat"
        elif low in ["longitude", "nav_lon"]:
            rename[name] = "lon"
    ds = ds.rename(rename)
    if ds["lon"].ndim == 1:
        ds = ds.assign_coords(lon=(ds["lon"] + 180) % 360 - 180).sortby("lon")
        ds = ds.sortby("lat")
    else:
        ds = ds.assign_coords(lon=(ds["lon"] + 180) % 360 - 180)
    return ds


def cut_native(ds: xr.Dataset, domain: dict, margin: float = DOMAIN_MARGIN) -> xr.Dataset:
    """restrict a native grid (1D or curvilinear) to the domain plus a margin, in the domain's longitude convention"""
    return dom.select(ds, domain, margin)


def global_mean(da: xr.DataArray, mask: xr.DataArray = None) -> xr.DataArray:
    """
    area-weighted (cos lat) mean over all non-time dimensions of a native field, ignoring missing values;
    `mask` (True = use, on the field's grid) restricts it, e.g. to the ocean
    """
    space = [d for d in da.dims if d != "time"]
    weights = np.cos(np.deg2rad(da["lat"])).fillna(0).clip(min=0)
    if weights.ndim == 1 and da["lat"].dims[0] in da.dims:
        weights = weights.broadcast_like(da.isel(time=0, drop=True))
    if mask is not None:
        weights = weights * mask.astype(float).values
    return da.weighted(weights).mean(space)


def global_mean_mask(cfg, kind, da: xr.DataArray):
    """mask for `global_mean` (None: all points) on the grid of `da`"""
    if kind in (None, False, True):
        return None
    from climate_scenarios_2050.preprocess import masks

    return masks.on_grid(masks.ocean_mask(cfg), da["lat"], da["lon"]).transpose(*[d for d in da.dims if d != "time"])


#---------------file discovery--------------#
@functools.lru_cache(maxsize=None)
def _discover_cached(cmip6_root: str, variable: str, scenario: str, full_period, table="Amon", folder=None) -> dict:
    return discover_members(cmip6_root, variable, scenario, full_period, table, folder)


def archive_folder(cfg, variable: str) -> str:
    """folder of a variable in the archive: `<variable>_<tag>` for the subsets of `download.subsets` (e.g. ua)"""
    tag = ((cfg.get("download") or {}).get("subsets") or {}).get(variable, {}).get("tag")
    return f"{variable}_{tag}" if tag else variable


def _spec(cfg, variable: str) -> tuple:
    """arguments of `_discover_cached` after the scenario: full period, table, archive folder"""
    full = cfg["cmip6"].get("full_period")
    return (tuple(full) if full else None, cmip6_table(cfg, variable), archive_folder(cfg, variable))


@functools.lru_cache(maxsize=None)
def _common_cached(cmip6_root: str, scenario: str, specs: tuple) -> frozenset:
    """(model, variant) complete in every variable of `specs` ((variable, full period, table, folder)); logs who drops out"""
    sets = {v: {(model, var) for model, variants in _discover_cached(cmip6_root, v, scenario, *rest).items() for var in variants}
            for v, *rest in specs}
    for v in [v for v, found in sets.items() if not found]:
        log.warning(f"{v} {scenario}: no complete member in the archive, not used for `cmip6.common_members`")
        del sets[v]
    if not sets:
        return frozenset()
    common = set.intersection(*sets.values())
    for v, found in sets.items():
        if found - common:
            log.info(f"{v} {scenario}: {len(found - common)} members left out, not complete in all of {list(sets)}")
    return frozenset(common)


def members(cfg, variable: str, scenario: str) -> dict:
    """
    members in the archive whose historical and scenario files cover `cmip6.full_period`; for the
    variables of `cmip6.common_members` only those complete in all of them (scanned once per process)
    """
    root = str(cfg["paths"]["inputs"]["cmip6"])
    own = _discover_cached(root, variable, scenario, *_spec(cfg, variable))
    common = list(cfg["cmip6"].get("common_members") or [])
    if variable not in common:
        return own
    keep = _common_cached(root, scenario, tuple((v, *_spec(cfg, v)) for v in common))
    out = {model: {var: files for var, files in variants.items() if (model, var) in keep} for model, variants in own.items()}
    return {model: variants for model, variants in out.items() if variants}


def model_key(variants: dict) -> str:
    """hash of a model's input: its members and the names of their files"""
    return config_hash({variant: {exp: sorted(Path(f).name for f in files) for exp, files in exps.items()} for variant, exps in sorted(variants.items())})


def members_hash(cfg, variable: str, scenario: str) -> str:
    """hash of the input of a variable/scenario: every model with the key of its file list"""
    return config_hash({model: model_key(variants) for model, variants in members(cfg, variable, scenario).items()})


def field_id(cfg, field: str, aggregation: str, scenario: str) -> str:
    """identifies the preprocessed field of a scenario (settings + archive state) for downstream hashes"""
    store, _ = field_store(cfg, field, aggregation)
    return f"{store.name}-{members_hash(cfg, field_variable(cfg, field), scenario)}"


def model_store(store: Path, scenario: str, model: str, variants: dict) -> Path:
    return store / scenario / f"{model}_{model_key(variants)}.zarr"


RANGE_PATTERN = re.compile(r"_(\d{4})(\d{2})-(\d{4})(\d{2})(?:_[^/]+)?\.nc$")


def _months(files, first_year, last_year) -> set:
    """months (year * 12 + month - 1) covered by the date ranges in the file names, within the given years"""
    out = set()
    for fl in files:
        m = RANGE_PATTERN.search(Path(fl).name)
        if m:
            y0, m0, y1, m1 = map(int, m.groups())
            out.update(range(max(y0 * 12 + m0 - 1, first_year * 12), min(y1 * 12 + m1 - 1, last_year * 12 + 11) + 1))
    return out


def discover_members(cmip6_root, variable: str, scenario: str, full_period=None, table: str = "Amon", folder: str = None) -> dict:
    """
    {model: {variant: {"historical": [files], scenario: [files]}}} for all members with both historical
    and scenario files (in `<experiment>/<model>/<folder or variable>/`). With `full_period` (first, last
    year) the historical files (until 2014) and the scenario files (from 2015) of a member must together
    cover every month of the period by their file names; other members (late historical starts such as
    EC-Earth3 r101+ from 1970, short scenario runs, gaps between files) are left out.
    """
    cmip6_root = Path(cmip6_root)
    found = {}
    for exp in ["historical", scenario]:
        for fl in sorted(cmip6_root.glob(f"{exp}/*/{folder or variable}/{variable}_{table}_*_{exp}_*.nc")):
            if fl.stat().st_size == 0:
                continue
            model = fl.parent.parent.name
            match = VARIANT_PATTERN.search(fl.name)
            if match is None:
                continue
            found.setdefault(model, {}).setdefault(match.group(1), {}).setdefault(exp, []).append(fl)
    members, incomplete = {}, []
    for model, variants in found.items():
        both = {var: files for var, files in variants.items() if "historical" in files and scenario in files}
        if full_period:
            first, last = full_period
            need = set(range(first * 12, last * 12 + 12))
            for var in list(both):
                have = _months(both[var]["historical"], first, HIST_END) | _months(both[var][scenario], HIST_END + 1, last)
                missing = need - have
                if missing:
                    incomplete.append(f"{model} {var} (from {min(missing) // 12})")
                    del both[var]
        if both:
            members[model] = dict(sorted(both.items()))
    if incomplete:
        log.info(f"{variable} {scenario}: {len(incomplete)} members left out, months of {full_period[0]}-{full_period[1]} "
                 f"missing: {', '.join(incomplete)}")
    return dict(sorted(members.items()))


def _join_time(parts):
    """
    concatenate along time on the grid of the first part: horizontal coordinates of a member's files
    can differ by rounding errors, which an outer join would turn into duplicated rows of NaNs
    """
    for part in parts[1:]:
        for name in ["lat", "lon"]:
            if part[name].shape != parts[0][name].shape or np.abs(part[name].values - parts[0][name].values).max() > GRID_TOLERANCE:
                raise ValueError(f"{name} differs between the files of a member")
    return xr.concat(parts, dim="time", data_vars="minimal", coords="minimal", compat="override", join="override")


def _open_runs(files, variable, years, chunks=None):
    """open and join the files of one experiment, restricted to calendar years `years` (inclusive)"""
    parts = []
    for fl in files:
        ds = xr.open_dataset(fl, use_cftime=True, decode_timedelta=False, chunks=chunks)
        # FGOALS/EC-Earth3-Veg: some files run beyond 2100 or overlap the other experiment
        year = ds["time"].dt.year
        keep = ((year >= years[0]) & (year <= years[1])).values
        if keep.any():
            parts.append(standardize_coords(ds[[variable]].isel(time=keep)))
    if not parts:
        return None
    return _join_time(parts).sortby("time")


def load_member_monthly(files: dict, variable: str, scenario: str, period, domain: dict, with_global_mean=False, mask_for=None):
    """
    joined monthly data (historical until 2014, scenario from 2015) of one member, cut to the
    domain; returns (DataArray, info) or (None, reason). With `with_global_mean`, info["global_mean"]
    holds the monthly global mean of the full native field (read lazily in chunks), restricted to
    `mask_for(native field)` if given.
    """
    chunks = {"time": 120} if with_global_mean else None
    hist = _open_runs(files["historical"], variable, (period[0], HIST_END), chunks)
    scen = _open_runs(files[scenario], variable, (HIST_END + 1, period[1]), chunks)
    if hist is None or scen is None:
        return None, "no data in period"
    ds = _join_time([hist, scen])
    # remove duplicated time steps (keep the first occurrence)
    _, first = np.unique(ds["time"].values, return_index=True)
    ds = ds.isel(time=np.sort(first))

    # check that the monthly series has no gaps between its first and last month
    time = ds["time"]
    months_since_start = (time.dt.year - time.dt.year[0]) * 12 + (time.dt.month - time.dt.month[0])
    if not np.array_equal(months_since_start.values, np.arange(time.size)):
        return None, "gaps in the monthly series"

    da = cut_native(ds, domain)[variable].load()
    if da.isnull().all(dim=[d for d in da.dims if d != "time"]).any():
        return None, "time steps without any data"
    info = {
        "first_month": str(time.values[0])[:7],
        "last_month": str(time.values[-1])[:7],
        "calendar": time.values[0].calendar,
    }
    if with_global_mean:
        native = ds[variable]
        info["global_mean"] = global_mean(native, mask_for(native) if mask_for else None).load()
    return da, info


#---------------regridding--------------#
def grid_key(da: xr.DataArray) -> str:
    return config_hash({"lat": np.round(da["lat"].values, 4).tolist(), "lon": np.round(da["lon"].values, 4).tolist()})


def get_regridder(src: xr.DataArray, tgt: xr.Dataset, method: str, weights_file: Path):
    src_grid = src.isel(time=0, drop=True).to_dataset(name="v") if "time" in src.dims else src.to_dataset(name="v")
    if method == "conservative":
        src_grid = add_cell_bounds(src_grid)
        tgt = add_cell_bounds(tgt)
    if weights_file.exists():
        return xe.Regridder(src_grid, tgt, method, weights=str(weights_file), ignore_degenerate=True, unmapped_to_nan=True)
    # ignore_degenerate: cells collapsing at the poles of grids with a pole row
    # unmapped_to_nan: target cells outside the source grid (north of a Gaussian grid's last row) would otherwise be 0
    regridder = xe.Regridder(src_grid, tgt, method, ignore_degenerate=True, unmapped_to_nan=True)
    weights_file.parent.mkdir(parents=True, exist_ok=True)
    regridder.to_netcdf(str(weights_file))
    return regridder


#---------------main routine--------------#
def field_store(cfg, field: str, aggregation: str) -> Path:
    """folder of the regridded archive of one field/aggregation (hash over all relevant settings)"""
    fcfg = cfg["preprocess"]["fields"][field]
    variable = field_variable(cfg, field)
    record = {
        "variable": variable,
        "aggregation": aggregation,
        "months": fcfg["aggregations"][aggregation],
        "method": fcfg["method"],
        "domain": fcfg["domain"],
        "grid_spacing": cfg["preprocess"]["grid_spacing"],
        "period": cfg["cmip6"]["period"],
    }
    # only recorded when not the default, so the stores of the other variables keep their hashes
    if fcfg.get("global_mean"):
        record["global_mean"] = fcfg["global_mean"] if fcfg["global_mean"] == "ocean" else True
        if fcfg["global_mean"] == "ocean":
            record["mask"] = "ERA5 SST"
    if cmip6_table(cfg, variable) != "Amon":
        record["table"] = cmip6_table(cfg, variable)
    if field != variable:
        record["field"] = field
    return Path(cfg["paths"]["outputs"]["cmip6_processed"]) / f"{field}_{aggregation}" / config_hash(record), record


def process_model(cfg, model: str, variants: dict, field: str, scenario: str, stores: dict):
    """process all members of one model and write one zarr per aggregation"""
    fcfg = cfg["preprocess"]["fields"][field]
    variable = field_variable(cfg, field)
    model_ids = load_model_ids()
    if model not in model_ids:
        raise KeyError(f"{model} has no id in configs/model_ids.yaml - append it there")
    mid = model_ids[model]
    # named after the model's file list: new members or files give a new store, old stores stay as they are
    out_files = {agg: model_store(stores[agg], scenario, model, variants) for agg in fcfg["aggregations"]}
    todo = [agg for agg, fl in out_files.items() if cfg["preprocess"]["overwrite"] or not fl.exists()]
    if not todo:
        return {"model": model, "status": "exists"}

    tgt = target_grid(cfg["preprocess"]["grid_spacing"], fcfg["domain"])
    weights_dir = Path(cfg["paths"]["outputs"]["cmip6_processed"]) / "regrid_weights"
    regridders = {}
    results = {agg: [] for agg in todo}
    with_gm = bool(fcfg.get("global_mean"))
    global_means = {agg: [] for agg in todo}
    masks_by_grid = {}

    def mask_for(native):
        key = grid_key(native)
        if key not in masks_by_grid:
            masks_by_grid[key] = global_mean_mask(cfg, fcfg.get("global_mean"), native)
        return masks_by_grid[key]

    report = []
    for variant, files in variants.items():
        try:
            da, info = load_member_monthly(files, variable, scenario, cfg["cmip6"]["period"], fcfg["domain"], with_gm, mask_for)
        except Exception as err:  # e.g. a corrupt file: drop the member, keep the run going
            da, info = None, f"could not read: {err}"
        if da is None:
            report.append({"model": model, "variant": variant, "kept": False, "reason": info})
            log.info(f"{model} {variant}: dropped ({info})")
            continue
        key = grid_key(da)
        if key not in regridders:
            weights_file = weights_dir / f"{model}_{key}_{fcfg['method']}_{config_hash([fcfg['domain'], cfg['preprocess']['grid_spacing']])}.nc"
            regridders[key] = get_regridder(da, tgt, fcfg["method"], weights_file)
        for agg in todo:
            seas = seasonal_mean(da, fcfg["aggregations"][agg])
            seas = regridders[key](seas, keep_attrs=True).astype("float32")
            info[f"nan_fraction_{agg}"] = round(float(seas.isnull().all("year").mean()), 4)
            if variable not in ZERO_ALLOWED and bool((seas == 0).any()):
                raise ValueError(f"{model} {variant}: exact zeros in the regridded {variable} field")
            seas = seas.expand_dims(member=[f"{mid}{variant}"])
            results[agg].append(seas)
            if with_gm:
                gm = seasonal_mean(info["global_mean"], fcfg["aggregations"][agg]).astype("float32")
                global_means[agg].append(gm.drop_vars([c for c in gm.coords if c != "year"]).expand_dims(member=[f"{mid}{variant}"]))
        info.pop("global_mean", None)
        report.append({"model": model, "variant": variant, "kept": True, **info})

    kept = [r for r in report if r["kept"]]
    if not kept:
        return {"model": model, "status": "no valid members", "report": report}

    for agg in todo:
        data = xr.concat(results[agg], dim="member", join="outer")
        data = data.assign_coords(
            model_id=("member", [mid] * data.sizes["member"]),
            model=("member", [model] * data.sizes["member"]),
            variant=("member", [r["variant"] for r in kept]),
            calendar=("member", [r["calendar"] for r in kept]),
            first_month=("member", [r["first_month"] for r in kept]),
            last_month=("member", [r["last_month"] for r in kept]),
        )
        ds = data.to_dataset(name=variable)
        if with_gm:
            ds[f"{variable}_global_mean"] = xr.concat(global_means[agg], dim="member", join="outer").reindex(member=ds.member, year=ds.year)
        ds.attrs = {"model": model, "scenario": scenario, "aggregation": agg, "months": str(fcfg["aggregations"][agg]), "regrid_method": fcfg["method"]}
        path = ensure_writable(out_files[agg], cfg, overwrite=cfg["preprocess"]["overwrite"])
        ds.chunk({"member": 1, "year": -1, "lat": -1, "lon": -1}).to_zarr(path, mode="w")
    return {"model": model, "status": "written", "report": report}


def _process_model_safe(cfg, model, variants, field, scenario, stores):
    """a failing model is reported at the end instead of stopping the other models"""
    try:
        return process_model(cfg, model, variants, field, scenario, stores)
    except Exception as err:
        return {"model": model, "status": f"FAILED: {err}"}


def run(cfg, fields=None, scenarios=None):
    """preprocess all models of the given fields/scenarios (defaults: the fields the configuration uses)"""
    fields = fields or used_fields(cfg)
    scenarios = scenarios or cfg["cmip6"]["scenarios"]
    for field in fields:
        fcfg = cfg["preprocess"]["fields"][field]
        variable = field_variable(cfg, field)
        stores = {}
        for agg in fcfg["aggregations"]:
            store, record = field_store(cfg, field, agg)
            if not (store / "config.yaml").exists():
                write_yaml(record, store / "config.yaml", cfg)
            stores[agg] = store
        for scenario in scenarios:
            found = members(cfg, variable, scenario)
            if not found:
                log.info(f"no {variable} data for {scenario}")
                continue
            log.info(f"{field} {scenario}: {len(found)} models, {sum(len(v) for v in found.values())} members (input {members_hash(cfg, variable, scenario)})")
            results = Parallel(n_jobs=cfg["resources"]["n_jobs"])(
                delayed(_process_model_safe)(cfg, model, variants, field, scenario, stores) for model, variants in found.items()
            )
            rows = [row for res in results for row in res.get("report", [])]
            if rows:
                report_file = stores[list(stores)[0]] / f"report_{scenario}_{pd.Timestamp.now():%Y%m%d%H%M%S}.csv"
                pd.DataFrame(rows).to_csv(ensure_writable(report_file, cfg), index=False)
            for res in results:
                log.info(f"{field} {scenario} {res['model']}: {res['status']}")


def _open_model(path) -> xr.Dataset:
    """
    one model's archive, with its member coordinates in memory and without the on-disk encoding: a
    string width kept in the encoding (<U10) would cut longer names of other models when the joined
    data is written (m12 -> m1, MPI-ESM1-2-HR -> MPI-ESM1-2).
    Scalar coordinates that only some models have (e.g. `height` of sfcWind) are dropped.
    """
    ds = xr.open_zarr(path)
    ds = ds.drop_vars([name for name in ds.coords if ds[name].ndim == 0])
    ds = ds.assign_coords({name: ds[name].load() for name in ds.coords if name not in ds.dims})
    return ds.reset_encoding()


def open_field(cfg, field: str, aggregation: str, scenario: str) -> xr.Dataset:
    """
    all models of one field/aggregation/scenario as one dataset (member, year, lat, lon), data variable
    named after the CMIP6 variable: the stores that match the current archive (models without a store,
    e.g. without valid members, are logged)
    """
    store, _ = field_store(cfg, field, aggregation)
    expected = {model: model_store(store, scenario, model, variants) for model, variants in members(cfg, field_variable(cfg, field), scenario).items()}
    files = [fl for fl in expected.values() if fl.exists()]
    missing = sorted(model for model, fl in expected.items() if not fl.exists())
    if missing:
        log.warning(f"{field} {aggregation} {scenario}: no preprocessed store for {len(missing)} of {len(expected)} models ({', '.join(missing)}); run the preprocessing if the archive changed")
    if not files:
        raise FileNotFoundError(f"no preprocessed {field} {aggregation} {scenario} in {store} for the current archive")
    return xr.concat([_open_model(fl) for fl in files], dim="member", join="outer")
