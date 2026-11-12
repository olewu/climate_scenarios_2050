"""
Harmonize and merge NAO indexes from HadSLP and ERA5
"""

import xarray as xr
import matplotlib.pyplot as plt

nao_e5 = xr.open_dataarray("/datalake/NS9873K/DATA/SFE/ERA5_processed/NAO_variability/internal-NAO-variability_f-None.nc")
nao_hadslp = xr.open_dataarray("/projects/NS9873K/owul/projects/climate_scenarios_2050/data/HadSLP/internal-NAO-variability_f-None.nc")

nao_e5_ref = nao_e5.sel(year=slice(nao_e5.year.min(),nao_hadslp.year.max()))
nao_hadslp_ref = nao_hadslp.sel(year=slice(nao_e5.year.min(),nao_hadslp.year.max()))

f,ax = plt.subplots()
nao_e5.plot(ax=ax)
nao_hadslp.plot(ax=ax)

had_mean = nao_hadslp_ref.mean()
had_std = nao_hadslp_ref.std()

era_mean = nao_e5_ref.mean()
era_std = nao_e5_ref.std()

print(had_mean.values,had_std.values,era_mean.values,era_std.values)

nao_e5_harm = (nao_e5 - era_mean)*(had_std/era_std) + had_mean

nao_e5_harm.plot(ax=ax,ls="dashed")

print(xr.corr(nao_e5_ref,nao_hadslp_ref).values)
print(((nao_e5_ref-nao_hadslp_ref)**2).mean().values**.5)

f,ax = plt.subplots()
ax.plot(nao_e5_harm.sel(year=slice(nao_e5.year.min(),nao_hadslp.year.max())).year,nao_e5_harm.sel(year=slice(nao_e5.year.min(),nao_hadslp.year.max())) - nao_hadslp_ref)
ax.set_title("Difference harmonized - HadSLP")

merged_nao = xr.concat([nao_hadslp.sel(year=slice(1979)),nao_e5_harm.sel(year=slice(1980,None))],dim="year")

merged_nao.to_netcdf("/datalake/NS9873K/DATA/CMIP6/wind_db/internal-NAO-variability-HadSLP-ERA5-merged-1980.nc")