"""
Configuration handling.

A run is configured by `configs/default.yaml` (all parameters with defaults), optional
override files (e.g. `configs/experiments/*.yaml`) and `configs/paths.yaml` (input and output
locations). Override files are deep-merged into the defaults in the given order.

Every output of a step is written to a folder named after a short hash of the parts of the
configuration the step depends on (including the hashes of the upstream steps). The resolved
configuration is stored next to the output (see `core.io.StepOutput`), so a hash can always be
traced back to its parameters.
"""

import copy
import hashlib
import json
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "configs"


def deep_merge(base: dict, override: dict) -> dict:
    """recursively merge `override` into a copy of `base`"""
    merged = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], val)
        else:
            merged[key] = copy.deepcopy(val)
    return merged


def load_yaml(path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def load_config(override_files=(), paths_file=None, default_file=None) -> dict:
    """
    load the default configuration, merge the override files and add the paths
    under the key `paths`
    """
    cfg = load_yaml(default_file or CONFIG_DIR / "default.yaml")
    for override_file in override_files:
        cfg = deep_merge(cfg, load_yaml(override_file))
    cfg["paths"] = load_yaml(paths_file or CONFIG_DIR / "paths.yaml")
    cfg = resolve_matching_window(cfg)
    cfg = resolve_region(cfg)
    validate(cfg)
    return cfg


def resolve_region(cfg: dict) -> dict:
    """field `domain: region` -> the domain of `region` (resolved before hashing)"""
    for fcfg in cfg.get("preprocess", {}).get("fields", {}).values():
        if fcfg.get("domain") == "region":
            fcfg["domain"] = copy.deepcopy(cfg["region"]["domain"])
    return cfg


def validate(cfg: dict):
    """consistency of the region, targets, predictor and observational datasets (clear errors early)"""
    targets, sources = cfg.get("targets", {}), cfg.get("obs_sources", {})
    fields = cfg.get("preprocess", {}).get("fields", {})
    for name in cfg.get("region", {}).get("targets", []):
        if name not in targets:
            raise ValueError(f"region.targets: {name} is not defined in `targets`")
        spec = targets[name]
        if spec["change"] not in ("relative", "absolute"):
            raise ValueError(f"targets.{name}.change must be relative or absolute")
        if spec["source"] == "file":
            if spec["file"] not in cfg["paths"]["inputs"].get("targets", {}):
                raise ValueError(f"targets.{name}: no path for {spec['file']} in paths.inputs.targets")
        elif spec["variable"] not in sources.get(spec["source"], {}).get("variables", {}):
            raise ValueError(f"targets.{name}: {spec['source']} does not provide {spec['variable']} (obs_sources)")
        if spec.get("cmip6") and spec["cmip6"] not in fields:
            raise ValueError(f"targets.{name}: no preprocess.fields entry {spec['cmip6']} (its CMIP6 field)")
        if spec.get("cmip6") and "annual" not in fields[spec["cmip6"]]["aggregations"]:
            raise ValueError(f"targets.{name}: preprocess.fields.{spec['cmip6']} needs an `annual` aggregation")
    if "obs" in cfg and cfg["obs"].get("target_variable") not in cfg.get("region", {}).get("targets", []):
        raise ValueError(f"obs.target_variable {cfg['obs'].get('target_variable')} is not in region.targets")
    pred = cfg.get("predictor")
    if pred:
        pf = predictor_field(cfg)
        if pf not in fields:
            raise ValueError(f"no preprocess.fields entry {pf} for the predictor")
        if field_variable(cfg, pf) != pred["variable"]:
            raise ValueError(f"preprocess.fields.{pf} has variable {field_variable(cfg, pf)}, the predictor {pred['variable']}")
        if pred["aggregation"] not in fields[pf]["aggregations"]:
            raise ValueError(f"predictor.aggregation {pred['aggregation']} is not an aggregation of preprocess.fields.{pf}")
        if pred.get("mask") not in (None, "ocean"):
            raise ValueError("predictor.mask must be null or ocean")
        for ds in pred["obs_datasets"]:
            if pred["variable"] not in sources.get(ds, {}).get("variables", {}):
                raise ValueError(f"obs_sources.{ds} does not provide the predictor variable {pred['variable']}")
        if pred.get("remove_global_mean") and not fields[pf].get("global_mean"):
            raise ValueError(f"predictor.remove_global_mean needs preprocess.fields.{pf}.global_mean (true or ocean)")
    for name, fcfg in fields.items():
        if fcfg.get("global_mean") not in (None, False, True, "ocean"):
            raise ValueError(f"preprocess.fields.{name}.global_mean must be true, false or ocean")
        merge = cfg.get("obs", {}).get("merge")
        if merge:
            for ds in merge["datasets"]:
                if ds not in pred["obs_datasets"]:
                    raise ValueError(f"obs.merge: {ds} is not in predictor.obs_datasets")
        names = [*pred["obs_datasets"], *(["merged"] if merge else [])]
        for key in ["matching_index", "training_index"]:
            if cfg.get("obs", {}).get(key) not in names:
                raise ValueError(f"obs.{key} must be one of {names}")


def active_targets(cfg: dict) -> list:
    return list(cfg["region"]["targets"])


def target_spec(cfg: dict, name: str) -> dict:
    return cfg["targets"][name]


def predictor_field(cfg: dict) -> str:
    """name of the predictor's entry in preprocess.fields (`predictor.field`, default: its variable)"""
    return cfg["predictor"].get("field") or cfg["predictor"]["variable"]


def field_variable(cfg: dict, field: str) -> str:
    """CMIP6 variable of a preprocess.fields entry (`variable`, default: the entry's name)"""
    return cfg["preprocess"]["fields"][field].get("variable", field)


def cmip6_table(cfg: dict, variable: str) -> str:
    return cfg.get("cmip6", {}).get("tables", {}).get(variable, "Amon")


def used_fields(cfg: dict) -> list:
    """fields (preprocess.fields entries) the configuration needs: the predictor's and the active targets' CMIP6 fields"""
    names = [predictor_field(cfg)]
    for name in active_targets(cfg):
        var = target_spec(cfg, name).get("cmip6")
        if var and var not in names:
            names.append(var)
    return names


def resolve_matching_window(cfg: dict) -> dict:
    """
    baseline `fit_window: matching` -> the analogue matching window (`analogues.window`), so the
    "fair" baselines always see as much of the past as the analogues. Resolved before hashing: the
    hash records the actual window (default 70, i.e. unchanged hashes).
    """
    baselines = cfg.get("baselines", {})
    for bcfg in [*baselines.get("ar1", {}).values(), baselines.get("sine_trend", {})]:
        if bcfg.get("fit_window") == "matching":
            bcfg["fit_window"] = cfg["analogues"]["window"]
    return cfg


def config_hash(obj, length: int = 8) -> str:
    """short, stable hash of a (json-serializable) configuration subset"""
    canonical = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha1(canonical.encode()).hexdigest()[:length]


def select(cfg: dict, sections) -> dict:
    """subset of the configuration a step depends on (missing sections are an error)"""
    return {sec: copy.deepcopy(cfg[sec]) for sec in sections}


def load_model_ids(path=None) -> dict:
    """fixed mapping model name -> short id (e.g. 'ACCESS-CM2' -> 'm0')"""
    return load_yaml(path or CONFIG_DIR / "model_ids.yaml")
