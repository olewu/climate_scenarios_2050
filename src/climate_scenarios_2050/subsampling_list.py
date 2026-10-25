from climate_scenarios_2050.config import *
import pandas as pd

all_models = pd.read_csv("/projects/NS9873K/owul/projects/climate_scenarios_2050/data/CMIP6_download_for_Climate_Futures.csv").model.unique()

Smith_best = [
    "UKESM1-0-LL",
    "KACE-1-0-G",
    "MPI-ESM1-2-LR",
    "MRI-ESM2-0",
    "MIROC-ES2L",
    "HadGEM3-GC31-LL",
    "CanESM5-CanOE",
    "CanESM5-1",
    "CanESM5",
    "CESM2-WACCM",
    "CESM2",
    "ACCESS-CM2",
    "GISS-E2-1-G",
    "NorESM2-LM",
]

Smith_intermediate = [
    "GFDL-ESM4",
    "ACCESS-ESM1-5",
    "MIROC6",
]

Smith_worst = [
    "CNRM-ESM2-1",
    "CNRM-CM6-1",
    "EC-Earth3-Veg-LR",
    "EC-Earth3-Veg",
    "EC-Earth3",
    "IPSL-CM6A-LR",
    "GISS-E2-2-G",
]

Smith_all_but_worst = [mod for mod in all_models if mod not in Smith_worst]

Smith_subsampling = {
    "Smith_incl_best" : Smith_best,
    "Smith_incl_best_plus" : Smith_best + Smith_intermediate,
    "Smith_excl_worst": Smith_all_but_worst,
}