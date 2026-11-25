"""
Safe output handling.

- Outputs can only be written below the output roots defined in `configs/paths.yaml`
  (`paths.outputs`), so input data can never be touched.
- Existing files are never overwritten unless `overwrite=True` is passed explicitly.
- Each step writes into `<root>/<step>/<config hash>/` together with a `config.yaml` that
  holds the resolved configuration subset and the upstream hashes.
"""

import contextlib
import logging
import shutil
from pathlib import Path

import yaml

from climate_scenarios_2050.core.config import config_hash, select

log = logging.getLogger(__name__)

# section -> {key: neutral value}: left out of the hash while they have this value
HASH_NEUTRAL = {
    "predictor": {"indices": None},
    "analogues": {"match_weights": None},
    "linear_model": {"predictors": None, "ridge": 0.0},
}


_READ_ONLY = False


@contextlib.contextmanager
def read_only():
    """within this block StepOutput.prepare() creates nothing and no file can be written (hash lookups)"""
    global _READ_ONLY
    previous, _READ_ONLY = _READ_ONLY, True
    try:
        yield
    finally:
        _READ_ONLY = previous


def output_roots(cfg) -> list:
    return [Path(p).resolve() for p in cfg["paths"]["outputs"].values()]


def ensure_writable(path, cfg, overwrite=False) -> Path:
    """
    check that `path` lies below one of the output roots and does not exist yet
    (or remove it if `overwrite` is set)
    """
    if _READ_ONLY:
        raise PermissionError(f"refusing to write {path}: read-only mode")
    path = Path(path).resolve()
    if not any(root == path or root in path.parents for root in output_roots(cfg)):
        raise PermissionError(f"refusing to write {path}: not below any output root in paths.yaml")
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass overwrite=True to replace it")
        log.warning(f"overwriting {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def write_netcdf(obj, path, cfg, overwrite=False, compress=False):
    """write a Dataset/DataArray to netcdf via a temporary file (no half-written outputs)"""
    path = ensure_writable(path, cfg, overwrite)
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        tmp.unlink()
    ds = obj if hasattr(obj, "data_vars") else obj.to_dataset()
    # encodings inherited from the input files must not apply here (a string width would cut longer strings)
    ds = ds.reset_encoding()
    encoding = None
    if compress:
        encoding = {name: {"zlib": True, "complevel": 4, "dtype": "float32"} for name in ds.data_vars if ds[name].dtype.kind == "f"}
    ds.to_netcdf(tmp, encoding=encoding)
    tmp.rename(path)
    return path


def write_yaml(obj, path, cfg, overwrite=False):
    path = ensure_writable(path, cfg, overwrite)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)
    return path


class StepOutput:
    """
    output location of one step for one configuration:
    `<paths.outputs[root_key]>/<step>/<hash>/`, hash over the config sections the step
    depends on, extra identifiers (e.g. scenario) and upstream hashes
    """

    def __init__(self, cfg, step, sections, upstream=None, extra=None, root_key="root", exclude=None):
        """`exclude`: {section: [keys]} of settings the step does not use (left out of the hash)"""
        self.cfg = cfg
        self.step = step
        config = select(cfg, sections)
        # settings added later that only enter the hash when they are used (older hashes stay valid)
        for section, neutral in HASH_NEUTRAL.items():
            for key, value in neutral.items():
                if section in config and key in config[section] and config[section][key] == value:
                    config[section].pop(key)
        for section, keys in (exclude or {}).items():
            for key in keys:
                config[section].pop(key, None)
        self.record = {
            "step": step,
            "config": config,
            "extra": extra or {},
            "upstream": upstream or {},
        }
        self.hash = config_hash(self.record)
        self.dir = Path(cfg["paths"]["outputs"][root_key]) / step / self.hash

    def prepare(self):
        """create the folder and write/compare the config record (nothing in `read_only()` mode)"""
        if _READ_ONLY:
            return self
        record_file = self.dir / "config.yaml"
        if record_file.exists():
            with open(record_file) as f:
                existing = yaml.safe_load(f)
            if existing != yaml.safe_load(yaml.safe_dump(self.record, sort_keys=False)):
                raise RuntimeError(f"hash collision or modified record in {record_file}")
        else:
            write_yaml(self.record, record_file, self.cfg)
        return self

    def path(self, name) -> Path:
        return self.dir / name

    def exists(self, name) -> bool:
        return self.path(name).exists()

    def __repr__(self):
        return f"StepOutput({self.step}, {self.hash}, {self.dir})"
