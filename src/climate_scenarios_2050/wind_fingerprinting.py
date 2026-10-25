import xarray as xr
import xeofs as xe
from climate_scenarios_2050.config import *
from climate_scenarios_2050.utils import make_season_group
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

scenario = "ssp245"

forced_signal_file = proj_base/f"tmp/psl_forced_annual_{scenario}.nc"

if forced_signal_file.exists():
    psl_forced_annual = xr.open_dataarray(forced_signal_file)
else:
    psl_simulation = xr.open_zarr(database_path/f"wind_db/psl_Amon_{scenario}_multimodel.zarr")["psl"]

    valid_members = ~psl_simulation.isnull().any(dim=("lon", "lat", "time")).compute()

    # cut out NAO domain:
    psl_simulation_NAO_domain = psl_simulation.sel(member=psl_simulation.member[valid_members]).sel(lon=NAO_domain["winter"]["longitude"],lat=NAO_domain["winter"]["latitude"])

    psl_mmem = psl_simulation_NAO_domain.mean("member")

    # aggregate forced signal to annual (does this make sense?)
    season = make_season_group(psl_mmem,months_in_season=[1,2,3,4,5,6,7,8,9,10,11,12])
    psl_forced_annual = psl_mmem.groupby(season).mean().sel(year=slice(1850,2098))

    # compute the above (takes ~10 min) and consider saving the result
    psl_forced_annual.to_netcdf(forced_signal_file)

# make anomalies:
psl_forced_annual_anomalies = psl_forced_annual - psl_forced_annual.sel(year=slice(1850,2015)).mean("year")

# set up EOF/PCA model
model = xe.single.EOF(n_modes=10, center=True, use_coslat=True)

# fit model:
model.fit(psl_forced_annual_anomalies.sel(year=slice(1850,2015)), dim="year")

PCs = model.transform(psl_forced_annual_anomalies)
