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

from climate_scenarios_2050.core.config import load_config

log = logging.getLogger("climate_scenarios_2050")

STEPS = ["preprocess", "split", "hindcast-nao", "evaluate-nao", "linear-model", "hindcast-wind", "evaluate-wind", "forecast", "update-index"]


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


def run_preprocess(cfg, args):
    from climate_scenarios_2050.preprocess import cmip6, obs
    from climate_scenarios_2050.preprocess.indices import compute

    parts = args.part or ["cmip6", "obs", "wind", "indices"]
    scenarios = args.scenario or None
    if "cmip6" in parts:
        cmip6.run(cfg, variables=args.variable or None, scenarios=scenarios)
    if "obs" in parts:
        for variable in args.variable or list(cfg["preprocess"]["fields"]):
            for dataset in cfg["preprocess"]["obs_datasets"]:
                if variable in obs.OBS_VARIABLES[dataset]:
                    obs.process_obs_field(cfg, dataset, variable)
    if "wind" in parts:
        from climate_scenarios_2050.split.run import available_wind_variables

        for wind_variable in available_wind_variables(cfg):
            obs.process_obs_wind(cfg, wind_variable)
    if "indices" in parts:
        compute.run(cfg, scenarios=scenarios)


def run_split(cfg, args):
    from climate_scenarios_2050.split import run as split

    split.run(cfg, scenarios=args.scenario or None)


def run_hindcast_nao(cfg, args):
    from climate_scenarios_2050.analogues import generate

    generate.run(cfg, scenarios=args.scenario or None)


def run_evaluate_nao(cfg, args):
    from climate_scenarios_2050.evaluate import nao

    nao.run(cfg, scenarios=args.scenario or None)


def run_linear_model(cfg, args):
    from climate_scenarios_2050.linear_model import fit

    fit.run(cfg, scenarios=args.scenario or None)


def run_hindcast_wind(cfg, args):
    from climate_scenarios_2050.methods import wind

    wind.run(cfg, scenarios=args.scenario or None)


def run_evaluate_wind(cfg, args):
    from climate_scenarios_2050.evaluate import wind

    wind.run(cfg, scenarios=args.scenario or None)


def run_forecast(cfg, args):
    from climate_scenarios_2050.forecast import forecast

    if args.init_year is None:
        raise SystemExit("forecast needs --init-year")
    windows = [tuple(int(v) for v in w.split("-")) for w in args.window] if args.window else None
    site = tuple(float(v) for v in args.site.split(",")) if args.site else None
    forecast.run(cfg, args.init_year, scenarios=args.scenario or None, wind_variable=args.wind, windows=windows, site=site, reference_wind=args.reference_wind, reference_end=args.reference_end)


def run_update_index(cfg, args):
    from climate_scenarios_2050.update import update_index

    update_index.run(cfg, files=args.era5_file or None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=STEPS)
    parser.add_argument("--config", action="append", default=[], help="override file(s), merged in order")
    parser.add_argument("--paths", default=None, help="alternative paths.yaml")
    parser.add_argument("--part", action="append", help="preprocess: cmip6 | obs | wind | indices (default: all)")
    parser.add_argument("--variable", action="append", help="preprocess: restrict to variable(s)")
    parser.add_argument("--scenario", action="append", help="restrict to scenario(s)")
    parser.add_argument("--init-year", type=int, help="forecast: initialization year")
    parser.add_argument("--window", action="append", help="forecast: target window a-b in lead years, e.g. 1-10 (default: target.windows)")
    parser.add_argument("--wind", help="forecast: wsp100 | energy (default: obs.wind_variable)")
    parser.add_argument("--site", help="forecast: lat,lon of a site for a table and figure")
    parser.add_argument("--reference-end", type=int, help="forecast: last year of the reference period (default: the init year, or the last year of the wind record if earlier)")
    parser.add_argument("--reference-wind", type=float, help="forecast: measured mean over the reference period, gives absolute changes at the site")
    parser.add_argument("--era5-file", action="append", help="update-index: monthly ERA5 SLP file(s) (default: all files in paths.inputs.era5_monthly)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, paths_file=args.paths)
    setup_logging(cfg, args.step)
    log.info(f"step {args.step}, overrides {args.config}")
    dask.config.set(scheduler="threads", num_workers=cfg["resources"]["dask_threads"])

    if args.step == "preprocess":
        run_preprocess(cfg, args)
    elif args.step == "split":
        run_split(cfg, args)
    elif args.step == "hindcast-nao":
        run_hindcast_nao(cfg, args)
    elif args.step == "evaluate-nao":
        run_evaluate_nao(cfg, args)
    elif args.step == "linear-model":
        run_linear_model(cfg, args)
    elif args.step == "hindcast-wind":
        run_hindcast_wind(cfg, args)
    elif args.step == "evaluate-wind":
        run_evaluate_wind(cfg, args)
    elif args.step == "forecast":
        run_forecast(cfg, args)
    elif args.step == "update-index":
        run_update_index(cfg, args)
    else:
        raise NotImplementedError(f"step {args.step} is not implemented yet")


if __name__ == "__main__":
    main()
