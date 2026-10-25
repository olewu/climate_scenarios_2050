import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from climate_scenarios_2050.config import *
from climate_scenarios_2050.filters import butter_lowpass_filter
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from pmdarima import auto_arima


#---------------Input Parameters---------------#
# use this observational dataset for the model:
ref_ds = "ERA5"

# 
rm_years = 8 

# split low from high frequency variability at this frequency:
co_per = 20
cutoff_lowfreq = 1/co_per

# model parameters for the ARIMA model to model the low-freq variability
pdq = (1,0,0) # (AR,0,MA)

# number of forecast steps
step = 35

# past/fitting window length:
win_len = 70 # 85 years available in ERA5

# number of synthetic NAO simulations
N = 500
spinup = 200

# number of members ("analogues" in the real predictions)
M_anlg = 100

#---------------Get the relevant NAO time series---------------#
if ref_ds == "ERA5":
    obs_ind_dir = Path("/datalake/NS9873K/DATA/SFE/ERA5_processed/NAO_variability")
else:
    obs_ind_dir = Path("/projects/NS9873K/owul/projects/climate_scenarios_2050/data/{:}".format(ref_ds))
index_int_var_obs = xr.open_dataarray(obs_ind_dir/"internal-NAO-variability_f-None.nc")


#---------------Make an ensemble of synthetic NAO time series---------------#
# fit an AR(1) process to the raw (!) NAO:
# only detrend:
trd_fll = np.polyfit(index_int_var_obs.year.values - 1850, index_int_var_obs.values,1) # estimate coefficients
trd_line = np.polyval(trd_fll,index_int_var_obs.year.values - 1850)
index_rm_dt = index_int_var_obs - trd_line

# AR(1) model:
mod_fll = ARIMA(index_rm_dt.values, order=pdq)  # Adjust order as needed
fitted_mod = mod_fll.fit()

# make N AR(1) simulations of raw NAO:
synth_index_notr = fitted_mod.simulate(
    nsimulations=spinup+len(index_rm_dt), # number of time steps to generate
    repetitions=N # how many different simulations to make
).squeeze()[spinup:]

synth_index = trd_line + synth_index_notr.T

# put into dataarray with correct dimensions and save:
synth_NAO = xr.Dataset(
    data_vars={
        "synthetic_NAO":(("syobs_member","year"),synth_index),
    },
    coords={
        "syobs_member" : ("syobs_member",np.arange(N)),
        "year" : ("year",index_rm_dt.year.values),
    },
)

synth_NAO.to_netcdf(base_path/f"owul/projects/climate_scenarios_2050/data/synthetic_ar1_{ref_ds}_nao.nc")

# iterate over reasonable values of start index:
start_indexes = range(61,len(index_int_var_obs)) # the 61=41 + 20 is from the length of the rolling window (41) applied to the previous 70 years in the analogue selection
init_years = index_int_var_obs.isel(year=slice(start_indexes[0],start_indexes[-1]+1)).year.values

# wrap into loop that can be executed in parallel
FORECAST = []; FORECAST_FULL = []; VERIFICATION = []; VALID_YEAR = []
FORECAST_ENS = []; FORECAST_ENS_FULL = [];
for ii,stix in enumerate(start_indexes):
    
    print(init_years[ii])

    #---------------Split data into training and testing set---------------#
    signal_train_da = index_int_var_obs.sel(year=slice(init_years[ii]-win_len,init_years[ii]))
    signal_train_da = signal_train_da.rolling(year=rm_years,center=True).mean().dropna("year")
    signal_test_da = index_int_var_obs.rolling(year=rm_years,center=True).mean().dropna("year").sel(year=slice(init_years[ii]-(rm_years//2)+1,init_years[ii]+step+rm_years//2-1))
    signal_train = signal_train_da.values
    signal_test = signal_test_da.values
    t_train = signal_train_da.year.values
    t_test_exvf = signal_test_da.year.values
    t_test = np.arange(t_train[-1],init_years[ii]+step+rm_years//2)

    # get a full training time series that sees all prior data:
    signal_train_da_full = index_int_var_obs.sel(year=slice(index_int_var_obs.year.isel(year=0),init_years[ii]-1))
    signal_train_da_full = signal_train_da_full.rolling(year=rm_years,center=True).mean().dropna("year")
    signal_train_full = signal_train_da_full.values
    t_train_full = signal_train_da_full.year.values

    #---------------Define the low-frequency variability part (modelled differently than high freq)---------------#
    # remove potential linear trend in the data:
    trend_coeff = np.polyfit(t_train - 1850,signal_train,deg=1)
    signal_train_trend = np.polyval(trend_coeff,t_train - 1850)
    
    signal_train_dt = signal_train - signal_train_trend

    trend_coeff_full = np.polyfit(t_train_full - 1850,signal_train_full,deg=1)
    signal_train_trend_full = np.polyval(trend_coeff_full,t_train_full - 1850)
    
    signal_train_dt_full = signal_train_full - signal_train_trend_full

    #---------------Model the low-frequency part---------------#
    # define high-freq signal as residual:
    result = adfuller(signal_train_dt)
    if result[1] > .05:
        print(f"Warning at init time {stix}; residuals are likely non-stationary!")
    
    # model low-freq var using ARIMA
    model = ARIMA(signal_train_dt, order=pdq)  # Adjust order as needed
    fitted_model = model.fit()
    # print(fitted_model.summary())

    # Full model:
    model_full = ARIMA(signal_train_dt_full, order=pdq)  # Adjust order as needed
    fitted_model_full = model_full.fit()


    #---------------Use the two models to make a PREDICTION---------------#
    # Forecast the residual
    forecast_lf = fitted_model.get_forecast(steps=step+rm_years-1).predicted_mean
    se, conf_95 = fitted_model.get_forecast(steps=step+rm_years-1).se_mean, fitted_model.get_forecast(steps=step+rm_years-1).conf_int(alpha=.05)
    # ensemble forecast:
    forecast_ens = fitted_model.simulate(nsimulations=step+rm_years-1, repetitions=M_anlg, anchor="end").squeeze()

    forecast = forecast_lf # + np.polyval(trend_coeff,t_test - 1850)

    forecast_full = fitted_model_full.get_forecast(steps=step+rm_years-1).predicted_mean
    se_full, conf_95_full = fitted_model_full.get_forecast(steps=step+rm_years-1).se_mean, fitted_model_full.get_forecast(steps=step+rm_years-1).conf_int(alpha=.05)
    # ensemble forecast:
    forecast_ens_full = fitted_model_full.simulate(nsimulations=step+rm_years-1, repetitions=M_anlg, anchor="end").squeeze()

    #---------------Plot a single example---------------#
    if stix == 110:
        plt.figure(figsize=(10, 5))
        plt.plot(t_test, forecast_ens.squeeze(),color="C3",lw=.5,alpha=.2)
        plt.plot(t_test, forecast_ens.squeeze()[:,45],color="C3",lw=1,alpha=1)
        plt.fill_between(t_test,conf_95[:,0],conf_95[:,1],color="C3",lw=.5,alpha=.1)
        plt.plot(t_train, signal_train, label="Original Signal (Training)",color="C0",lw=2)
        # plt.plot(t_train, signal_train_lp, label="Low-frequency Signal (de-trended)",color="C0",ls="dashed",alpha=.3)
        # plt.plot(t_train[co_per//2:-co_per//2], signal_train_lp[co_per//2:-co_per//2], label="Low-frequency Signal (de-trended)",color="C0",ls="dashed")
        plt.plot(t_test_exvf, signal_test, label="Original Signal (Testing)",color="C1",lw=2)
        # plt.plot(t_test, forecast_lf, label="Forecast (low-freq, no trend)",ls="dashed",color="C1")
        plt.plot(t_test, forecast, label="Forecast",color="C1")
        plt.plot(t_test, forecast_full, label="Forecast (full)",color="C1",ls="dashed")
        plt.vlines(init_years[ii]-.5,-150,150,color='k')
        plt.legend(ncol=3)
        plt.grid()
        plt.title("Example Prediction")
    
    #---------------Collect Predictions & Verification---------------#
    # Note that not all prediction time steps will have evaluation data available!
    verification = np.zeros_like(t_test)*np.nan
    verification[:len(signal_test)] = signal_test
    VALID_YEAR.append(t_test)
    VERIFICATION.append(verification)
    FORECAST.append(forecast)
    FORECAST_FULL.append(forecast_full)
    FORECAST_ENS.append(forecast_ens)
    FORECAST_ENS_FULL.append(forecast_ens_full)


# wrap into xarray dataset:
fcvf = xr.Dataset(
    data_vars={
        "forecast":(("init_year","lead_time"),FORECAST),
        "forecast_full":(("init_year","lead_time"),FORECAST_FULL),
        "forecast_ens":(("init_year","lead_time","member"),FORECAST_ENS),
        "forecast_ensfull":(("init_year","lead_time","member"),FORECAST_ENS_FULL),
        "verification":(("init_year","lead_time"),VERIFICATION),
    },
    coords={
        "init_year":("init_year",init_years),
        "lead_time":("lead_time",np.arange(-rm_years//2+1,step+rm_years//2)),
        "valid_year":(("init_year","lead_time"),VALID_YEAR),
        "member":(("member"),np.arange(100))
    }
)

xr.Dataset(
    data_vars={
        "forecast":(("init_year","lead_time","member"),FORECAST_ENS),
        "forecast_full":(("init_year","lead_time","member"),FORECAST_ENS_FULL),
        "verification":(("init_year","lead_time"),VERIFICATION),
    },
    coords={
        "init_year":("init_year",init_years),
        "lead_time":("lead_time",np.arange(-rm_years//2+1,step+rm_years//2)),
        "valid_year":(("init_year","lead_time"),VALID_YEAR),
        "member":(("member"),np.arange(100))
    }
).to_netcdf(base_path/f"owul/projects/climate_scenarios_2050/data/AR1_hindcast/AR1_{ref_ds}_100member_NAO_hindcast.nc")

# Correlation by lead time:
corr_by_lt = xr.corr(fcvf.forecast,fcvf.verification,dim="init_year")
corr_by_lt_full = xr.corr(fcvf.forecast_full,fcvf.verification,dim="init_year")

# RMSE:
rmse_by_lt = ((fcvf.forecast - fcvf.verification)**2).mean(dim="init_year",skipna=True)**.5
rmse_by_lt_full = ((fcvf.forecast_full - fcvf.verification)**2).mean(dim="init_year",skipna=True)**.5

# Plot scores:
f_corr,ax_corr = plt.subplots()
corr_by_lt.plot(ax=ax_corr,label="correlation")
corr_by_lt_full.plot(ax=ax_corr,label="correlation (full)")
(rmse_by_lt/fcvf.verification.std("init_year")).plot(ax=ax_corr,label="RMSE")
(rmse_by_lt_full/fcvf.verification.std("init_year")).plot(ax=ax_corr,label="RMSE (full)")
plt.hlines(0,0,35,color='k')
ax_corr.legend()
ax_corr.set_xlim([0,step])

# Quick visualization:
fig,ax = plt.subplots()
fcvf.forecast_full.plot(x="init_year",hue="lead_time",add_legend=False,color='grey',ax=ax);
fcvf.verification.sel(lead_time=1).rolling(init_year=8).mean().plot(color='k',ax=ax)

# # Detrend:
# trend = np.polyfit(t_train, signal_train, 1)
# detrended_signal = signal_train - np.polyval(trend, t_train)



# # Plot
# plt.figure(figsize=(10, 5))
# plt.plot(t_train, signal_train, label="Original Signal (Training)")
# plt.plot(t_train, detrended_signal, label="Detrended Signal")
# plt.plot(t_test, signal_test, label="Original Signal (Testing)")
# plt.legend()
# plt.title("Detrending")
# plt.show()




# from statsmodels.tsa.holtwinters import ExponentialSmoothing
# from statsmodels.tsa.statespace.sarimax import SARIMAX

# # Example: SARIMA(1, 0, 1)(0, 0, 0, 1) (non-seasonal only)
# # model = SARIMAX(detrended_signal, order=(1, 0, 1), seasonal_order=(0, 0, 0, 1))
# # Fit exponential smoothing
# model = ExponentialSmoothing(
#     detrended_signal,
#     trend="add",  # or "mul" for multiplicative
#     seasonal=None,  # no seasonality
# )
# fitted_model = model.fit()
# print(fitted_model.summary())

# forecast = fitted_model.forecast(steps=step)
# forecast_full = fitted_model.forecast(steps=step) + np.polyval(trend, t_test)

# plt.figure(figsize=(10, 5))
# plt.plot(t_train, signal_train, label="Original Signal (Training)")
# plt.plot(t_train, detrended_signal, label="Detrended Signal")
# plt.plot(t_test, signal_test, label="Original Signal (Testing)")
# plt.plot(t_test, forecast, label="Prediction")
# plt.plot(t_test, forecast_full, label="Prediction + Trend")
# plt.legend()
# plt.title("Detrending")
# plt.show()
