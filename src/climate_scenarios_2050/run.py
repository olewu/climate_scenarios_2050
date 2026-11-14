"""
Command line entry point for all steps:

    python -m climate_scenarios_2050.run <step> [--config override.yaml ...] [options]

e.g. `pixi run preprocess --config configs/experiments/eof2.yaml --part indices`.
"""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import dask

from climate_scenarios_2050.core.config import load_config, predictor_field

log = logging.getLogger("climate_scenarios_2050")

STEPS = ["cmip6-inventory", "cmip6-download", "preprocess", "split", "hindcast-index", "evaluate-index", "linear-model", "hindcast-target", "evaluate-target", "forecast", "update-index"]


def setup_logging(cfg, step):
    log_dir = Path(cfg["paths"]["outputs"]["root"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{step}_{datetime.now():%Y%m%d-%H%M%S}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    log.info(f"log file: {log_file}")


def run_cmip6_inventory(cfg, args):
    from climate_scenarios_2050.cmip_retrieval import inventory

    inventory.run(cfg, variables=args.variable or None, experiments=args.scenario or None)


def run_cmip6_download(cfg, args):
    from climate_scenarios_2050.cmip_retrieval import download

    download.run(cfg, target=args.target, inventory=args.inventory, variables=args.variable or None,
                 experiments=args.scenario or None, max_datasets=args.max_datasets)


def run_preprocess(cfg, args):
    from climate_scenarios_2050.preprocess import cmip6, obs
    from climate_scenarios_2050.preprocess.indices import compute

    parts = args.part or ["cmip6", "obs", "targets", "indices"]
    scenarios = args.scenario or None
    if "cmip6" in parts:
        cmip6.run(cfg, fields=args.variable or None, scenarios=scenarios)
    if "obs" in parts:
        # observed fields of the predictor (the targets have their own part)
        for dataset in cfg["predictor"]["obs_datasets"]:
            obs.process_obs_field(cfg, dataset, predictor_field(cfg))
    if "targets" in parts:
        from climate_scenarios_2050.split.run import available_target_variables

        for target_variable in available_target_variables(cfg):
            obs.process_obs_target(cfg, target_variable)
    if "indices" in parts:
        compute.run(cfg, scenarios=scenarios)


def run_split(cfg, args):
    from climate_scenarios_2050.split import run as split

    split.run(cfg, scenarios=args.scenario or None)


def run_hindcast_index(cfg, args):
    from climate_scenarios_2050.analogues import generate

    generate.run(cfg, scenarios=args.scenario or None)


def run_evaluate_index(cfg, args):
    from climate_scenarios_2050.evaluate import index

    index.run(cfg, scenarios=args.scenario or None)


def run_linear_model(cfg, args):
    from climate_scenarios_2050.linear_model import fit

    fit.run(cfg, scenarios=args.scenario or None)


def run_hindcast_target(cfg, args):
    from climate_scenarios_2050.methods import target

    target.run(cfg, scenarios=args.scenario or None)


def run_evaluate_target(cfg, args):
    from climate_scenarios_2050.evaluate import target

    target.run(cfg, scenarios=args.scenario or None)


def run_forecast(cfg, args):
    from climate_scenarios_2050.forecast import forecast

    if args.init_year is None:
        raise SystemExit("forecast needs --init-year")
    windows = [tuple(int(v) for v in w.split("-")) for w in args.window] if args.window else None
    site = tuple(float(v) for v in args.site.split(",")) if args.site else None
    forecast.run(cfg, args.init_year, scenarios=args.scenario or None, target_variable=args.target_variable, windows=windows, site=site, reference_value=args.reference_value, reference_end=args.reference_end)


def run_update_index(cfg, args):
    from climate_scenarios_2050.update import update_index

    update_index.run(cfg, files=args.era5_file or None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--config", action="append", default=[], help="override file(s), merged in order")
    parser.add_argument("--paths", default=None, help="alternative paths.yaml")
    parser.add_argument("--part", action="append", help="preprocess: cmip6 | obs | targets | indices (default: all)")
    parser.add_argument("--variable", action="append", help="preprocess (cmip6 part): restrict to fields (preprocess.fields); cmip6-*: to CMIP6 variables")
    parser.add_argument("--scenario", action="append", help="restrict to scenario(s) (cmip6-*: experiments, incl. historical)")
    parser.add_argument("--target", help="cmip6-download: archive root to write to (default: paths.inputs.cmip6)")
    parser.add_argument("--inventory", help="cmip6-download: inventory folder with plan.csv (default: the latest)")
    parser.add_argument("--max-datasets", type=int, help="cmip6-download: only the first n planned datasets (testing)")
    parser.add_argument("--init-year", type=int, help="forecast: initialization year")
    parser.add_argument("--window", action="append", help="forecast: target window a-b in lead years, e.g. 1-10 (default: target.windows)")
    parser.add_argument("--target-variable", help="forecast: a key of paths.inputs.targets, e.g. wsp100 | energy (default: obs.target_variable)")
    parser.add_argument("--site", help="forecast: lat,lon of a site for a table and figure")
    parser.add_argument("--reference-end", type=int, help="forecast: last year of the reference period (default: the init year, or the last year of the target record if earlier)")
    parser.add_argument("--reference-value", type=float, help="forecast: measured mean over the reference period, gives absolute changes at the site")
    parser.add_argument("--era5-file", action="append", help="update-index: monthly ERA5 SLP file(s) (default: all files in paths.inputs.era5_monthly)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, paths_file=args.paths)
    setup_logging(cfg, args.step)
    log.info(f"step {args.step}, overrides {args.config}")
    dask.config.set(scheduler="threads", num_workers=cfg["resources"]["dask_threads"])

    if args.step == "cmip6-inventory":
        run_cmip6_inventory(cfg, args)
    elif args.step == "cmip6-download":
        run_cmip6_download(cfg, args)
    elif args.step == "preprocess":
        run_preprocess(cfg, args)
    elif args.step == "split":
        run_split(cfg, args)
    elif args.step == "hindcast-index":
        run_hindcast_index(cfg, args)
    elif args.step == "evaluate-index":
        run_evaluate_index(cfg, args)
    elif args.step == "linear-model":
        run_linear_model(cfg, args)
    elif args.step == "hindcast-target":
        run_hindcast_target(cfg, args)
    elif args.step == "evaluate-target":
        run_evaluate_target(cfg, args)
    elif args.step == "forecast":
        run_forecast(cfg, args)
    elif args.step == "update-index":
        run_update_index(cfg, args)
    else:
        raise NotImplementedError(f"step {args.step} is not implemented yet")


if __name__ == "__main__":
    main()
