"""
Compare magnitude of forced signal varaibility vs model uncertainty vs internal variability
in the NAO
"""

import xarray as xr
from climate_scenarios_2050.config import *
import matplotlib.pyplot as plt
import numpy as np
# import plotly

Nyr = 8
poly_order = 2
REF_START, REF_END = 1995, 2014
scen_names = ["ssp119", "ssp126", "ssp245", "ssp370", "ssp585"]

def load_model_index(scen_names, rolling_mean_years = 1):

    full_ind = {}; forced_response = {}
    for scenario in scen_names:
        internal = xr.open_dataarray(database_path/f"internal/internal-NAO-variability_hist-{scenario}_f-None.nc")
        forced = xr.open_dataarray(database_path/f"forced/forced-NAO-variability_hist-{scenario}_f-None.nc")
        forced_response[scenario] = forced.rolling(year=rolling_mean_years,center=True).mean()
        full_ind[scenario] = (internal.groupby("m_id") + forced).rolling(year=rolling_mean_years,center=True).mean()

    return full_ind, forced_response


def compute_split_poylfit(full_ind, REF_START, REF_END, poly_order = 4):

    fit_model_list = []   # per-scenario (year, m_id) forced-response
    I_model_list   = []   # per-scenario (m_id,) internal variance

    for scen_name, da in full_ind.items():
        # da: dims (year, member), coord m_id(member)
        yr0, yr1 = da.year.values.min(), da.year.values.max()
        da = da.assign_coords(x=("year", (da.year.values - yr0) / (yr1 - yr0)))
        da = da.swap_dims({"year": "x"})

        coefs = da.polyfit("x", deg=poly_order, skipna=True).polyfit_coefficients
        fit = xr.polyval(da.x, coefs).where(da.notnull().any("x"))
        fit = fit.swap_dims({"x": "year"}); da = da.swap_dims({"x": "year"})

        # --- explicit reference: baseline = fitted curve averaged over REF years ---
        baseline = fit.sel(year=slice(REF_START, REF_END)).mean("year", skipna=True)  # (member,)
        fit_anom = fit - baseline
        da_anom  = da - baseline     # keep raw anomalies consistent, for residuals

        # restrict to desired years (e.g. drop historical period)
        fit_anom = fit_anom.sel(year=slice(2015, 2080))
        da_anom  = da_anom.sel(year=slice(2015, 2080))

        # residual after removing forced signal (polynomial fit)
        resid = da_anom - fit_anom
        n = resid.notnull().sum("year")
        # variance of residuals over all years --> internal variability in each member (!)
        I_run = ((resid**2).sum("year", skipna=True) / (n - (poly_order + 1))).where(n > poly_order + 1)

        fit_model = fit.groupby("m_id").mean("member")        # (year, m_id) — unique models only
        I_model   = I_run.groupby("m_id").mean("member")      # (m_id,)

        fit_model_list.append(fit_model.expand_dims(scen=[scen_name]))
        I_model_list.append(I_model.expand_dims(scen=[scen_name]))

    fit_model_all = xr.concat(fit_model_list, dim="scen", join="outer")  # (year, m_id, scen)
    I_model_all   = xr.concat(I_model_list,   dim="scen", join="outer")  # (m_id, scen)

    I = I_model_all.mean(("m_id", "scen"), skipna=True)
    M = fit_model_all.var("m_id", ddof=1, skipna=True).mean("scen")
    S = fit_model_all.mean("m_id", skipna=True).var("scen", ddof=1)

    return I, M, S, I + M + S

def compute_split_mmem(full_ind, forced_response, REF_START, REF_END):

    # --- 1. Merge your projected forced-response arrays (dims: year, m_id) per scenario ---
    fit_model_list = [fr.expand_dims(scen=[s]) for s, fr in forced_response.items()]
    fit_model_all = xr.concat(fit_model_list, dim="scen", join="outer")     # (year, m_id, scen), raw units

    # --- 2. Explicit reference period, for M/S only (baseline cancels out of I, see below) ---
    baseline = fit_model_all.sel(year=slice(REF_START, REF_END)).mean("year", skipna=True)  # (m_id, scen)
    fit_anom = (fit_model_all - baseline).sel(year=slice(2015, 2100))

    # --- 3. Internal variability: raw member minus ITS MODEL'S forced curve
    I_model_list = []
    for scen_name, da in full_ind.items():
        # da: (year, member), coord m_id(member), spans historical+future
        da = da.sel(year=slice(2015, 2100))
        forced_raw = fit_model_all.sel(scen=scen_name).sel(m_id=da.m_id)   # broadcasts m_id -> member
        resid = da - forced_raw                                            # baseline cancels here, so use raw fit_model_all

        n = resid.notnull().sum("year")
        I_run = ((resid**2).sum("year", skipna=True) / (n - 1)).where(n > 1)   # 1 dof: shared model curve, no per-member fit
        I_model = I_run.groupby("m_id").mean("member")
        I_model_list.append(I_model.expand_dims(scen=[scen_name]))

    I_model_all = xr.concat(I_model_list, dim="scen", join="outer")

    # --- 4. I, M, S ---
    I = I_model_all.mean(("m_id", "scen"), skipna=True)
    M = fit_anom.var("m_id", ddof=1, skipna=True).mean("scen")
    S = fit_anom.mean("m_id", skipna=True).var("scen", ddof=1)

    return I, M, S, I + M + S

full_ind, forced_response = load_model_index(scen_names, rolling_mean_years = Nyr)
I_poly, M_poly, S_poly, T_poly = compute_split_poylfit(full_ind, REF_START, REF_END, poly_order = poly_order)
I, M, S, T = compute_split_mmem(full_ind, forced_response, REF_START, REF_END)

# model uncertainty might be small by design since every model's forced response is
# obtained by projecting the SMEM onto the MMEM, meaning the forced response across models
# is highly correlated resulting in quite low variance across them
# how model uncertainty and internal variability are split depends quite a bit on how
# the forced response is estimated (no method is perfect), but the important point is 
# that scenario uncertainty is not super dependent on the method (MMEM or polynomial), 
# which is the main point here: scenario uncertainty is not what we need to care about
# most for the NAO when looking at the next ~25 years!

frac = xr.concat([I + 0 * M, M, S], dim="source").assign_coords(source=["internal", "model", "scenario"]) / T
frac_poly = xr.concat([I_poly + 0 * M_poly, M_poly, S_poly], dim="source").assign_coords(source=["internal", "model", "scenario"]) / T_poly

f,ax = plt.subplots()
ax.stackplot(frac_poly.year,frac_poly.values)
ax.set_xlim([2015,2080])
ax.set_ylim([0,1])
ax.annotate("internal variability",(2050,.4),color="white")
ax.annotate("model uncertainty",(2060,.8),color="white")
ax.annotate("scenario uncertainty",(2060,.95),color="white")

f,ax = plt.subplots()
ax.stackplot(frac.year,frac.values, labels=frac.source.values)
ax.set_xlim([2015,2075])
ax.set_xticks([2020,2030,2040,2050,2060,2070])
ax.set_xticklabels(["5", "15", "25", "35", "45", "55"])
ax.set_xlabel("years into future")
ax.set_ylabel("farction of variance")
ax.set_ylim([0,1])
ax.set_title(f"Future NAO variance split ({Nyr}-year average)")
ax.legend(loc=4)
# ax.annotate("internal variability",(2035,.4),color="white")
# ax.annotate("model uncertainty",(2055,.8),color="white")
# ax.annotate("scenario uncertainty",(2055,.95),color="white")
