"""
Comparison pages (`pixi run gallery --config ... [--name NAME]`): exports the results of one configuration
("experiment") for the static pages in `<figures>/gallery/` and registers the experiment there under a name.

The pages draw everything in the browser from the exported numbers (no image files):
- `gallery/data/evaluate_target/<hash>_<parts>.bin`: all maps of one target-evaluation folder, int16
  (value / SCALE, NaN = -32768, clipped to +-3.2767) one after the other, then the significance masks as bit
  sets; `<hash>_<parts>.json`: grid, byte offsets of every map, field statistics of the significance test.
  `<parts>`: `s` (scores only) or `ss` (scores and significance), plus the export version (`_v2`: with
  `msss_m1`). An export is never rewritten: once the significance test of a folder exists, or the export
  format changes, a new export is written next to the old one.
- `gallery/data/linear_model/<hash>.bin/.json`: the linear model's maps in the same format (correlation and
  RMSE skill given the observed index; target change per 1 std of the window-mean index, per predictor).
- `gallery/data/evaluate_index/<hash>_<parts>_v2.json`: the index evaluation (skill per lead time, window
  scores incl. the CRPS, null test, hindcast overview, CRPS per lead) as plain JSON; `<parts>`: letters of the files present
  (`_v2`: with the CRPS; the older exports without a suffix are kept).
- `gallery/data/partition/<hash>.json`: the variance partition of one index or target region (`partition.py`).
- `gallery/data/coast_<id>.json`: Natural Earth 50 m coastlines clipped to a grid's extent.
- `gallery/experiments.json`: the registry, experiment name -> config files and the exports per case.
- the page files themselves (copied from `gallery/static/` in the package when they changed).

Folders are shared between experiments (same hash -> same export).
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from climate_scenarios_2050.core.config import active_targets, config_hash
from climate_scenarios_2050.core.io import ensure_writable

log = logging.getLogger(__name__)

SCALE = 1e-4
MISSING = -32768
STATIC = Path(__file__).parent / "static"
SCORES = ["corr", "crpss", "msss_m1"]
TARGET_EXPORT_VERSION = "v2"   # v2: msss_m1 (1 - MSE / MSE of M1)
INDEX_EXPORT_VERSION = "v2"    # v2: CRPS of the window changes
FIELD_STATS = ["field_mean", "p_field_mean", "area_fraction_positive", "p_area_fraction", "null_field_mean_q95"]


def gallery_dir(cfg) -> Path:
    return Path(cfg["paths"]["outputs"]["figures"]) / "gallery"


def _to_int16(values: np.ndarray) -> np.ndarray:
    out = np.round(np.clip(values, -3.2767, 3.2767) / SCALE)
    return np.where(np.isfinite(values), out, MISSING).astype("<i2")


def coastlines(cfg, lat: np.ndarray, lon: np.ndarray) -> str:
    """coastline file for the extent of a grid (written once), name relative to gallery/data"""
    margin = 2.0
    box = [float(lon.min()) - margin, float(lon.max()) + margin, float(lat.min()) - margin, float(lat.max()) + margin]
    name = f"coast_{config_hash([round(b, 2) for b in box])}.json"
    fl = gallery_dir(cfg) / "data" / name
    if fl.exists():
        return name
    from cartopy.io import shapereader
    from shapely.geometry import box as shp_box

    shift = 360.0 if lon.max() > 180 else 0.0   # grid in 0..360
    clip = shp_box(box[0], box[2], box[1], box[3])
    lines = []
    for geom in shapereader.Reader(shapereader.natural_earth("50m", "physical", "coastline")).geometries():
        for off in ([0.0, shift] if shift else [0.0]):
            from shapely.affinity import translate

            part = translate(geom, xoff=off).intersection(clip)
            if part.is_empty:
                continue
            for seg in getattr(part, "geoms", [part]):
                if seg.geom_type == "LineString" and len(seg.coords) > 1:
                    lines.append([[round(x, 2), round(y, 2)] for x, y in seg.coords])
    ensure_writable(fl, cfg)
    fl.write_text(json.dumps({"box": box, "lines": lines}, separators=(",", ":")))
    return name


class Packer:
    """collects the maps/masks of one export file and writes it with its JSON index (never over an existing one)"""

    def __init__(self):
        self.chunks, self.offset = [], 0

    def add(self, data: np.ndarray) -> list:
        b = data.tobytes()
        self.chunks.append(b)
        entry = [self.offset, len(b)]
        self.offset += len(b)
        return entry

    def write(self, cfg, folder: Path, stem: str, index: dict):
        ensure_writable(folder / f"{stem}.bin", cfg)
        tmp = folder / f"{stem}.bin.tmp"
        with open(tmp, "wb") as f:
            for b in self.chunks:
                f.write(b)
        tmp.rename(folder / f"{stem}.bin")
        ensure_writable(folder / f"{stem}.json", cfg)
        (folder / f"{stem}.json").write_text(json.dumps(index, separators=(",", ":")))


def _grid_index(cfg, lat, lon, **extra) -> dict:
    return {"scale": SCALE, "missing": MISSING, "shape": [len(lat), len(lon)], "lat": np.round(lat, 4).tolist(),
            "lon": np.round(lon, 4).tolist(), "coast": coastlines(cfg, lat, lon), "maps": {}, **extra}


def export_target_evaluation(cfg, out) -> str:
    """export of one evaluate_target folder (StepOutput), name relative to gallery/data; existing exports are kept"""
    has_sig, has_m1 = out.exists("significance.nc"), out.exists("significance_m1.nc")
    parts = ("ss" if has_sig else "s") + ("1" if has_m1 else "")
    stem = f"{out.hash}_{parts}_{TARGET_EXPORT_VERSION}"
    folder = gallery_dir(cfg) / "data" / "evaluate_target"
    if (folder / f"{stem}.json").exists():
        return f"evaluate_target/{stem}"
    scores = xr.open_dataset(out.path("scores.nc"))
    # mean-squared-error skill against M1 (zero change, the operational method): 1 - MSE(m) / MSE(M1)
    if "M1" in scores.fc_method.values:
        scores["msss_m1"] = 1 - scores["rmse"] ** 2 / scores["rmse"].sel(fc_method="M1", drop=True) ** 2
    sig = xr.open_dataset(out.path("significance.nc")) if has_sig else None
    lat, lon = scores["latitude"].values, scores["longitude"].values
    index = _grid_index(cfg, lat, lon, hash=out.hash, parts=parts, sig={}, field={},
                        extra={k: v for k, v in out.record["extra"].items() if k in ("scenario", "target_variable")},
                        units=out.record["extra"].get("spec", {}).get("change", ""),
                        fdr_q=float(sig.attrs["fdr_q"]) if sig is not None and "fdr_q" in sig.attrs else None)
    pack = Packer()

    layers = [(var, scores[var]) for var in SCORES if var in scores]
    if sig is not None:
        layers += [(str(s), sig["value"].sel(statistic=s)) for s in sig.statistic.values]
    for name, da in layers:
        for target in da.target.values:
            for method in da.fc_method.values:
                for window in da.window.values:
                    values = da.sel(target=target, fc_method=method, window=window).transpose("latitude", "longitude").values.astype("float64")
                    if not np.isfinite(values).any():
                        continue
                    index["maps"][f"{name}|{target}|{method}|{window}"] = pack.add(_to_int16(values))
    if sig is not None:
        for s in sig.statistic.values:
            for target in sig.target.values:
                for method in sig.fc_method.values:
                    for window in sig.window.values:
                        sub = sig.sel(statistic=s, target=target, fc_method=method, window=window)
                        key = f"{s}|{target}|{method}|{window}"
                        if key not in index["maps"]:
                            continue
                        mask = sub["significant"].transpose("latitude", "longitude").values.astype(bool)
                        index["sig"][key] = pack.add(np.packbits(mask.ravel()))
                        stats = {k: float(sub[k]) for k in FIELD_STATS if k in sub}
                        index["field"][key] = {k: (round(v, 5) if np.isfinite(v) else None) for k, v in stats.items()}
    if has_m1:
        # whether a method beats M1 (paired block bootstrap): masks and field tests under "m1:<statistic>|..."
        m1 = xr.open_dataset(out.path("significance_m1.nc"))
        for st in m1.statistic.values:
            for target in m1.target.values:
                for method in m1.fc_method.values:
                    for window in m1.window.values:
                        key = f"{st}|{target}|{method}|{window}"
                        if key not in index["maps"]:
                            continue
                        sub = m1.sel(statistic=st, target=target, fc_method=method, window=window)
                        mask = sub["significant"].transpose("latitude", "longitude").values.astype(bool)
                        index["sig"]["m1:" + key] = pack.add(np.packbits(mask.ravel()))
                        stats = {k: float(sub[k]) for k in ("field_mean", "p_field_mean") if k in sub}
                        index["field"]["m1:" + key] = {k: (round(v, 5) if np.isfinite(v) else None) for k, v in stats.items()}
    pack.write(cfg, folder, stem, index)
    log.info(f"exported {out.dir} -> {folder / stem}.bin ({pack.offset / 1e6:.0f} MB, {len(index['maps'])} maps)")
    return f"evaluate_target/{stem}"


def _round(values, digits=4):
    """array -> nested lists with None for NaN (JSON)"""
    arr = np.asarray(values, dtype="float64")
    out = np.where(np.isfinite(arr), np.round(arr, digits), np.nan).tolist()
    def clean(x):
        return [clean(v) for v in x] if isinstance(x, list) else (None if x != x else x)
    return clean(out)


def export_linear_model(cfg, out, scenario, target_variable) -> str:
    """maps of one linear_model folder (hindcast scenarios: skill given the observed index; all: change per 1 std)"""
    from climate_scenarios_2050.core.config import target_spec
    from climate_scenarios_2050.plotting.linear_model import UNITS, window_index_std

    folder = gallery_dir(cfg) / "data" / "linear_model"
    stem = out.hash
    if (folder / f"{stem}.json").exists():
        return f"linear_model/{stem}"
    from climate_scenarios_2050.preprocess import obs

    skill = xr.open_dataset(out.path("perfect_index_skill.nc"))
    params = xr.open_dataset(out.path("hindcast.nc"))
    target = obs.open_obs_target(cfg, target_variable)
    units = target.attrs.get("units") or UNITS.get(target_variable, "units of the target")
    relative = target_spec(cfg, target_variable)["change"] == "relative"
    lat, lon = skill["latitude"].values, skill["longitude"].values
    predictors = [str(p) for p in params.predictor.values]
    index = _grid_index(cfg, lat, lon, hash=out.hash, extra={"scenario": scenario, "target_variable": target_variable},
                        predictors=predictors, units=units, limits={}, index_std={}, n_inits=int(params.sizes["init_year"]))
    pack = Packer()

    def add(key, da, limit=None):
        values = da.transpose("latitude", "longitude").values.astype("float64")
        if not np.isfinite(values).any():
            return
        index["maps"][key] = pack.add(_to_int16(values))
        if limit is not None:
            index["limits"][key.split("|")[0]] = max(index["limits"].get(key.split("|")[0], 0), limit)

    windows = [str(w) for w in skill.window.values]
    for w in windows:
        add(f"corr|{w}", skill["corr"].sel(window=w), 1.0)
        add(f"nrmse|{w}", skill["nrmse"].sel(window=w), 1.0)
    mean_target = target.mean("year")
    for predictor in predictors:
        std = window_index_std(cfg, scenario, windows, predictor)
        index["index_std"][predictor] = {w: round(float(std.sel(window=w)), 4) for w in windows}
        change = params["slope"].sel(predictor=predictor).mean("init_year") * std
        for name, field in [("change", change)] + ([("change_pct", change / mean_target * 100)] if relative else []):
            lim = float(np.nanquantile(np.abs(field.values), 0.98))
            for w in windows:
                add(f"{name}:{predictor}|{w}", field.sel(window=w).drop_vars("predictor", errors="ignore"), round(lim, 4))
    pack.write(cfg, folder, stem, index)
    log.info(f"exported {out.dir} -> {folder / stem}.bin ({pack.offset / 1e6:.0f} MB, {len(index['maps'])} maps)")
    return f"linear_model/{stem}"


def export_index_evaluation(cfg, out, scenario, index_name) -> str:
    """one evaluate_index folder as JSON (small): continuous and window skill, null test, hindcast overview"""
    files = {"c": "continuous.nc", "w": "windows.nc", "n": "null.nc", "o": "hindcast_overview.nc", "k": "continuous_crps.nc"}
    parts = "".join(k for k, f in files.items() if out.exists(f))
    stem = f"{out.hash}_{parts}_{INDEX_EXPORT_VERSION}"
    folder = gallery_dir(cfg) / "data" / "evaluate_index"
    if (folder / f"{stem}.json").exists():
        return f"evaluate_index/{stem}"
    data = {"hash": out.hash, "scenario": scenario, "index": index_name}
    if "c" in parts:
        ds = xr.open_dataset(out.path(files["c"]))
        data["continuous"] = {"lead": ds.lead.values.tolist(), "products": [str(p) for p in ds["product"].values],
                              **{v: _round(ds[v].values) for v in ds.data_vars if not v.endswith("_n")}}
    if "w" in parts:
        ds = xr.open_dataset(out.path(files["w"]))
        data["windows"] = {"window": [str(w) for w in ds.window.values], "products": [str(p) for p in ds["product"].values],
                           **{v: _round(ds[v].values) for v in ["corr", "nrmse", "crps", "crpss", "corr_ci_low", "corr_ci_high", "sign_agreement"] if v in ds}}
    if "n" in parts:
        ds = xr.open_dataset(out.path(files["n"]))
        data["null"] = {v: _round(ds[v].values) for v in ds.data_vars}
    if "k" in parts:
        ds = xr.open_dataset(out.path(files["k"]))
        data["continuous_crps"] = {"lead": ds.lead.values.tolist(), "products": [str(p) for p in ds["product"].values],
                                   **{v: _round(ds[v].values) for v in ds.data_vars}}
    if "o" in parts:
        ds = xr.open_dataset(out.path(files["o"]))
        data["overview"] = {"init_year": ds.init_year.values.tolist(), "lead": ds.lead.values.tolist(),
                            "analogue_mean": _round(ds["analogue_mean"].values, 3), "year": ds.year.values.tolist(),
                            "observed": _round(ds["observed"].values, 3), "attrs": {k: str(v) for k, v in ds.attrs.items()}}
    ensure_writable(folder / f"{stem}.json", cfg)
    (folder / f"{stem}.json").write_text(json.dumps(data, separators=(",", ":")))
    log.info(f"exported {out.dir} -> {folder / stem}.json")
    return f"evaluate_index/{stem}"


def export_partition(cfg, out) -> str:
    """one partition folder as JSON (small): variances, fractions and scenario means per window length and lead"""
    folder = gallery_dir(cfg) / "data" / "partition"
    if (folder / f"{out.hash}.json").exists():
        return f"partition/{out.hash}"
    ds = xr.open_dataset(out.path("partition.nc"))
    src = [str(s) for s in ds.source.values]
    data = {"hash": out.hash, **{k: (v.item() if hasattr(v, "item") else v) for k, v in ds.attrs.items()},
            "reference_length": int(ds.attrs["init"]) - int(str(ds.attrs["reference_period"]).split("-")[0]) + 1,
            "window_length": ds.window_length.values.tolist(), "lead": ds.lead.values.tolist(),
            "centre_year": ds["centre_year"].values.tolist(), "scenarios": [str(s) for s in ds.scenario.values],
            "n_models": ds["n_models"].values.tolist(), "n_members": ds["n_members"].values.tolist(),
            "variance": {s: _round(ds["variance"].sel(source=s).values, 6) for s in src},
            "fraction": {s: _round(ds["fraction"].sel(source=s).values) for s in src},
            "total": _round(ds["total"].values, 6),
            "scenario_mean": {str(s): _round(ds["scenario_mean"].sel(scenario=s).values) for s in ds.scenario.values}}
    ensure_writable(folder / f"{out.hash}.json", cfg)
    (folder / f"{out.hash}.json").write_text(json.dumps(data, separators=(",", ":")))
    log.info(f"exported {out.dir} -> {folder / out.hash}.json")
    return f"partition/{out.hash}"


def install_pages(cfg):
    """copy the page files (only new or changed ones; they are code, not results)"""
    target = gallery_dir(cfg)
    for src in STATIC.iterdir():
        dst = target / src.name
        if dst.exists() and dst.read_bytes() == src.read_bytes():
            continue
        if not dst.exists():
            ensure_writable(dst, cfg)
        shutil.copyfile(src, dst)
        log.info(f"page {dst}")


def update_registry(cfg, name: str, configs, entry: dict, replace=False):
    fl = gallery_dir(cfg) / "experiments.json"
    registry = json.loads(fl.read_text()) if fl.exists() else {"experiments": {}}
    old = registry["experiments"].get(name)
    if old and old.get("configs") != configs and not replace:
        raise SystemExit(f"experiment name {name!r} is registered with other configs {old.get('configs')}; choose another --name or pass --replace")
    # cases without a result under the current settings keep their earlier export, marked as stale (e.g. after a
    # change of the evaluation settings, until the step has been rerun)
    for step, cases in (old or {}).items():
        if isinstance(cases, dict) and step in entry and isinstance(entry[step], dict) and step != "info":
            for key, case in cases.items():
                if key not in entry[step] and isinstance(case, dict) and "export" in case:
                    entry[step][key] = {**case, "stale": True}
    registry["experiments"][name] = {"configs": configs, "registered": datetime.now().isoformat(timespec="seconds"), **entry}
    tmp = fl.with_name(fl.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    ensure_writable(tmp, cfg)
    tmp.write_text(json.dumps(registry, indent=1))
    tmp.rename(fl)
    return fl


def run(cfg, config_files, name=None, replace=False):
    """exports every existing index evaluation, linear model and target evaluation of the configuration"""
    from climate_scenarios_2050.core.config import hindcast_indices
    from climate_scenarios_2050.evaluate.index import evaluation_output
    from climate_scenarios_2050.evaluate.target import target_evaluation_output
    from climate_scenarios_2050.linear_model.fit import linear_model_output

    configs = [Path(c).name for c in config_files]
    name = name or ("+".join(Path(c).stem for c in config_files) or "default")
    entry = {"evaluate_index": {}, "linear_model": {}, "evaluate_target": {}, "partition": {}}
    for scenario in cfg["cmip6"]["scenarios_hindcast"]:
        for index_name in hindcast_indices(cfg):
            out = evaluation_output(cfg, scenario, index_name)
            if out.exists("continuous.nc") or out.exists("windows.nc"):
                entry["evaluate_index"][f"{scenario}|{index_name}"] = {"hash": out.hash, "export": export_index_evaluation(cfg, out, scenario, index_name)}
            else:
                log.info(f"{scenario} {index_name}: no index evaluation yet ({out.hash})")
        for tv in active_targets(cfg):
            out = linear_model_output(cfg, scenario, tv)
            if out.exists("perfect_index_skill.nc") and out.exists("hindcast.nc"):
                entry["linear_model"][f"{scenario}|{tv}"] = {"hash": out.hash, "export": export_linear_model(cfg, out, scenario, tv)}
            else:
                log.info(f"{scenario} {tv}: no linear model hindcast yet ({out.hash})")
            out = target_evaluation_output(cfg, scenario, tv)
            if out.exists("scores.nc"):
                entry["evaluate_target"][f"{scenario}|{tv}"] = {"hash": out.hash, "export": export_target_evaluation(cfg, out)}
            else:
                log.info(f"{scenario} {tv}: no target evaluation yet ({out.hash})")
    from climate_scenarios_2050 import partition

    for label, out in partition.outputs(cfg):
        key = "|".join(label.split(" "))   # index|nao, target|wsp100|scandinavia
        if out.exists("partition.nc"):
            ds_attrs = xr.open_dataset(out.path("partition.nc")).attrs
            entry["partition"][key] = {"hash": out.hash, "export": export_partition(cfg, out), "cmip6": ds_attrs.get("cmip6", "")}
        else:
            log.info(f"{label}: no partition yet ({out.hash}, pixi run partition)")
    from climate_scenarios_2050.core.config import lm_predictors, match_weights, primary_index, target_spec

    # for the method descriptions on the pages
    entry["info"] = {"primary_index": primary_index(cfg), "lm_predictors": lm_predictors(cfg), "match_weights": match_weights(cfg),
                     "predictor_lag": cfg["target"]["predictor_lag"], "reference_length": cfg["target"]["reference_length"],
                     "matching_index": cfg["obs"]["matching_index"], "training_index": cfg["obs"]["training_index"],
                     "M": cfg["analogues"]["M"], "obs_fit": (cfg["evaluation"].get("obs_methods") or {}).get("fit", "up_to_init"), "targets": {tv: {**target_spec(cfg, tv), "forced": target_spec(cfg, tv).get("forced", "index")} for tv in active_targets(cfg)}}
    if not any(v for k, v in entry.items() if k != "info"):
        raise SystemExit("nothing to export: no evaluation of this configuration exists yet")
    install_pages(cfg)
    fl = update_registry(cfg, name, configs, entry, replace)
    log.info(f"experiment {name!r}: " + ", ".join(f"{len(v)} {k}" for k, v in entry.items() if k != "info") + f"; registry {fl}")
    return name
