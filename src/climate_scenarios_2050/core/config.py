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
    cfg = resolve_jet_mask(cfg)
    validate(cfg)
    return cfg


def resolve_region(cfg: dict) -> dict:
    """field `domain: region` -> the domain of `region` (resolved before hashing)"""
    for fcfg in cfg.get("preprocess", {}).get("fields", {}).values():
        if fcfg.get("domain") == "region":
            fcfg["domain"] = copy.deepcopy(cfg["region"]["domain"])
    return cfg


def resolve_jet_mask(cfg: dict) -> dict:
    """`predictor.jet.mask` (a file in configs/masks): its checksum is added, so a changed mask changes the hashes"""
    pred = cfg.get("predictor") or {}
    for opts in [pred.get("jet"), *[(i or {}).get("jet") for i in (pred.get("indices") or {}).values()]]:
        if opts and opts.get("mask"):
            fl = CONFIG_DIR / "masks" / f"{opts['mask']}.nc"
            if not fl.exists():
                raise ValueError(f"predictor.jet.mask: no file {fl}")
            opts["mask_checksum"] = hashlib.sha256(fl.read_bytes()).hexdigest()[:16]
    return cfg


def validate(cfg: dict):
    """consistency of the region, targets, predictor and observational datasets (clear errors early)"""
    targets, sources = cfg.get("targets", {}), cfg.get("obs_sources", {})
    fields = cfg.get("preprocess", {}).get("fields", {})
    c6 = cfg.get("cmip6", {})
    full = c6.get("full_period")
    if full is not None and not (len(full) == 2 and all(isinstance(y, int) for y in full) and full[0] <= full[1]):
        raise ValueError(f"cmip6.full_period must be [first year, last year], got {full}")
    if "min_scenario_end" in c6:
        raise ValueError("cmip6.min_scenario_end was replaced by cmip6.full_period (the last year is the required scenario end)")
    for name in cfg.get("region", {}).get("targets", []):
        if name not in targets:
            raise ValueError(f"region.targets: {name} is not defined in `targets`")
        spec = targets[name]
        if spec["change"] not in ("relative", "absolute"):
            raise ValueError(f"targets.{name}.change must be relative or absolute")
        if spec.get("forced", "index") not in ("index", "cmip6"):
            raise ValueError(f"targets.{name}.forced must be index or cmip6")
        if spec.get("forced") == "cmip6" and not spec.get("cmip6"):
            raise ValueError(f"targets.{name}.forced: cmip6 needs a CMIP6 variable (targets.{name}.cmip6)")
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
    if cfg.get("predictor"):
        validate_indices(cfg)
        for name in index_names(cfg):
            _validate_predictor(index_cfg(cfg, name), sources, fields)
    for name, fcfg in fields.items():
        if fcfg.get("global_mean") not in (None, False, True, "ocean"):
            raise ValueError(f"preprocess.fields.{name}.global_mean must be true, false or ocean")


def _validate_predictor(cfg: dict, sources: dict, fields: dict):
    pred = cfg["predictor"]
    if pred:
        pf = predictor_field(cfg)
        if pf not in fields:
            raise ValueError(f"no preprocess.fields entry {pf} for the predictor")
        if field_variable(cfg, pf) != pred["variable"]:
            raise ValueError(f"preprocess.fields.{pf} has variable {field_variable(cfg, pf)}, the predictor {pred['variable']}")
        if pred["aggregation"] not in fields[pf]["aggregations"]:
            raise ValueError(f"predictor.aggregation {pred['aggregation']} is not an aggregation of preprocess.fields.{pf}")
        point = pred.get("sign_reference_point")
        if point is not None and point.get("sign", "positive") not in ("positive", "negative"):
            raise ValueError("predictor.sign_reference_point.sign must be positive or negative")
        if pred.get("method") == "jet":
            opts = pred.get("jet") or {}
            if opts.get("quantity", "latitude") not in ("latitude", "speed"):
                raise ValueError("predictor.jet.quantity must be latitude or speed")
            if opts.get("refine", "parabola") not in ("parabola", "none"):
                raise ValueError("predictor.jet.refine must be parabola or none")
            if not 0 < float(opts.get("min_valid", 0.8)) <= 1:
                raise ValueError("predictor.jet.min_valid must be in (0, 1]")
        if pred.get("method") == "amv":
            if not 0 < float((pred.get("amv") or {}).get("min_coverage", 0.5)) <= 1:
                raise ValueError("predictor.amv.min_coverage must be in (0, 1]")
        if pred.get("mask") not in (None, "ocean"):
            raise ValueError("predictor.mask must be null or ocean")
        for ds in pred["obs_datasets"]:
            if pred["variable"] not in sources.get(ds, {}).get("variables", {}):
                raise ValueError(f"obs_sources.{ds} does not provide the predictor variable {pred['variable']}")
        if pred.get("remove_global_mean") and not fields[pf].get("global_mean"):
            raise ValueError(f"predictor.remove_global_mean needs preprocess.fields.{pf}.global_mean (true or ocean)")
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


#---------------several indices--------------#
def index_names(cfg: dict) -> list:
    """names of the indices; the first is the primary index"""
    indices = cfg["predictor"].get("indices")
    return list(indices) if indices else [cfg["predictor"]["name"]]


def primary_index(cfg: dict) -> str:
    return index_names(cfg)[0]


def index_cfg(cfg: dict, name: str = None) -> dict:
    """
    configuration of one index: the predictor block with the index's overrides and `indices: null`, so
    that every per-index step (patterns, indices, split, index evaluation) hashes exactly like a
    single-index configuration of that index. Without `predictor.indices` the configuration itself.
    """
    name = name or primary_index(cfg)
    pred = cfg["predictor"]
    indices = pred.get("indices")
    if not indices:
        if name != pred["name"]:
            raise KeyError(f"index {name} is not configured (predictor.name is {pred['name']})")
        return cfg
    if name not in indices:
        raise KeyError(f"index {name} is not in predictor.indices ({list(indices)})")
    out = dict(cfg)
    base = {k: v for k, v in pred.items() if k != "indices"}
    out["predictor"] = deep_merge(base, indices[name] or {})
    out["predictor"]["name"] = name
    out["predictor"]["indices"] = None
    return out


def match_weights(cfg: dict) -> dict:
    """{index: weight} of the analogue matching, weights summing to one"""
    weights = cfg["analogues"].get("match_weights") or {primary_index(cfg): 1.0}
    weights = {k: float(v) for k, v in weights.items() if float(v) > 0}
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def lm_predictors(cfg: dict) -> list:
    """indices of the linear model"""
    return list(cfg["linear_model"].get("predictors") or [primary_index(cfg)])


def hindcast_indices(cfg: dict) -> list:
    """indices the analogue hindcast provides: the primary, the matched ones and the linear model's"""
    names = [primary_index(cfg), *match_weights(cfg), *lm_predictors(cfg)]
    return list(dict.fromkeys(names))


def validate_indices(cfg: dict):
    names = index_names(cfg)
    for key, used in [("analogues.match_weights", match_weights(cfg)), ("linear_model.predictors", lm_predictors(cfg))]:
        unknown = [n for n in used if n not in names]
        if unknown:
            raise ValueError(f"{key}: {unknown} not in the configured indices {names}")
    if not match_weights(cfg):
        raise ValueError("analogues.match_weights: no positive weight")
    if float(cfg["linear_model"].get("ridge") or 0) < 0:
        raise ValueError("linear_model.ridge must be >= 0")


def predictor_field(cfg: dict) -> str:
    """name of the predictor's entry in preprocess.fields (`predictor.field`, default: its variable)"""
    return cfg["predictor"].get("field") or cfg["predictor"]["variable"]


def field_variable(cfg: dict, field: str) -> str:
    """CMIP6 variable of a preprocess.fields entry (`variable`, default: the entry's name)"""
    return cfg["preprocess"]["fields"][field].get("variable", field)


def cmip6_table(cfg: dict, variable: str) -> str:
    return cfg.get("cmip6", {}).get("tables", {}).get(variable, "Amon")


def used_fields(cfg: dict) -> list:
    """fields (preprocess.fields entries) the configuration needs: the indices' and the active targets' CMIP6 fields"""
    names = list(dict.fromkeys(predictor_field(index_cfg(cfg, name)) for name in index_names(cfg)))
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
