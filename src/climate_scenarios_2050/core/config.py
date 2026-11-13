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
    return resolve_matching_window(cfg)


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
