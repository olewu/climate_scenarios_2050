import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from climate_scenarios_2050.config import *
from climate_scenarios_2050.filters import butter_lowpass_filter
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from scipy.optimize import curve_fit
from scipy.fft import fft, fftfreq
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from pmdarima import auto_arima

#---------------Cyclostationary + trend model for low-frequency variability---------------#
def combined_model(
        t,
        A1, phi1, f1,
        # A2, phi2, f2,
        # A3, phi3, f3,
        m, b
    ):
    return (A1 * np.sin(2 * np.pi * f1 * t + phi1) +
            # A2 * np.sin(2 * np.pi * f2 * t + phi2) +
            # A3 * np.sin(2 * np.pi * f3 * t + phi3) +
            m * t + b)  # Linear trend + sine waves

def estimate_initial_frequencies(signal, t):
    # Compute FFT
    n = len(signal)
    # remove the mean (DC component) and the lowest frequency since it represents more the trend
    fft_result = fft(signal)[2:n//2]
    frequencies = fftfreq(n, d=t[1] - t[0])[2:n//2]

    # Get the top 2 frequencies
    amplitudes = np.abs(fft_result)
    top_indices = np.argsort(amplitudes)[-2:]
    initial_frequencies = np.abs(frequencies[top_indices])

    return initial_frequencies


#---------------Input Parameters---------------#
# use this observational dataset for the model:
ref_ds = "HadSLP"

# 
rm_years = 8 

# split low from high frequency variability at this frequency:
co_per = 20
cutoff_lowfreq = 1/co_per

# model parameters for the ARIMA model for high-frequ variability
pdq = (0,0,1)

# number of forecast steps
step = 35

# past/fitting window length:
win_len = 70 # 85 years available in ERA5


#---------------Get the relevant NAO time series---------------#
obs_ind_dir = Path("/projects/NS9873K/owul/projects/climate_scenarios_2050/data/{:}".format(ref_ds))
index_int_var_obs = xr.open_dataarray(obs_ind_dir/"internal-NAO-variability_f-None.nc")
index_int_var_obs = index_int_var_obs.rolling(year=rm_years).mean().dropna("year")

# iterate over reasonable values of start index:
start_indexes = range(win_len,len(index_int_var_obs))
init_years = index_int_var_obs.isel(year=slice(start_indexes[0],start_indexes[-1]+1)).year.values

# wrap into loop that can be executed in parallel
FORECAST = []; VERIFICATION = []; VALID_YEAR = []
for ii,stix in enumerate(start_indexes):
    
    print(init_years[ii])

    #---------------Split data into training and testing set---------------#
    t_train = index_int_var_obs.isel(year=slice(stix-win_len,stix)).year.values
    t_test_exvf = index_int_var_obs.isel(year=slice(stix,stix+step)).year.values
    t_test = np.arange(init_years[ii],init_years[ii]+step)
    signal_train = index_int_var_obs.isel(year=slice(stix-win_len,stix)).values
    signal_test = index_int_var_obs.isel(year=slice(stix,stix+step)).values


    #---------------Define the low-frequency variability part (modelled differently than high freq)---------------#
    # Isolate the low-frequency component with a filter:
    signal_train_lp = butter_lowpass_filter(signal_train,cutoff=cutoff_lowfreq,fs=1)

    #---------------Model the high-frequency part---------------#
    # define high-freq signal as residual:
    signal_train_res = signal_train - signal_train_lp
    result = adfuller(signal_train_res)
    if result[1] > .05:
        print(f"Warning at lead time: {stix}; residuals are likely non-stationary!")
    
    # model residuals using ARIMA
    model = ARIMA(signal_train_res, order=pdq)  # Adjust order as needed
    fitted_model = model.fit()
    # print(fitted_model.summary())

    
    #---------------Model the low-frequency part---------------#

    # Get some reasonable initial conditions for the curve fitting
    # trend estimate
    trend_coeffs = np.polyfit(t_train - 1850, signal_train_lp, 1)

    signal_std = signal_train_lp.std()
    init_freq = estimate_initial_frequencies(signal_train_lp,t_train)

    p0 = [
        .2*signal_std, 0, init_freq[0],
        # .2*signal_std, 0, init_freq[1],
        # 0.5, 1/30, 0,
        *trend_coeffs
    ]  # Initial guess

    trend_range = (signal_train_lp.max() - signal_train_lp.min())/len(signal_train_lp)
    intercept_range = 1.5*abs(signal_train_lp).max()

    bounds = (
        [
            0, -np.pi, 0.01,
            # 0, -np.pi, 0.01,
            -trend_range, -intercept_range
        ],  # Lower bounds
        [
            2*signal_std, np.pi, 0.1,
            # 2*signal_std, np.pi, 0.1,
            trend_range, intercept_range
        ]   # Upper bounds
    )

    # consider removing some of the edges since the filtering could introduce spurious behaviour there
    try:
        popt, _ = curve_fit(combined_model, t_train[co_per//2:-co_per//2] - 1850, signal_train_lp[co_per//2:-co_per//2], p0=p0, bounds=bounds, maxfev=5000, method='trf')
    except:
        print("failed to fit sine model, using previous years model")
    
    #---------------Use the two models to make a PREDICTION---------------#
    # Forecast the residual
    residual_forecast = fitted_model.get_forecast(steps=step).predicted_mean
    
    # Predict using the fitted model parameters:
    low_freq_forecast = combined_model(t_test - 1850, *popt)

    # Obtain full forecast as sum of the high- and low- frequency models
    forecast = low_freq_forecast + residual_forecast

    #---------------Plot a single example---------------#
    if stix == 110:
        plt.figure(figsize=(10, 5))
        plt.plot(t_train, signal_train, label="Original Signal (Training)",color="C0",lw=2)
        plt.plot(t_train, signal_train_lp, label="Low-frequency Signal",color="C0",ls="dashed",alpha=.3)
        plt.plot(t_train[co_per//2:-co_per//2], signal_train_lp[co_per//2:-co_per//2], label="Low-frequency Signal",color="C0",ls="dashed")
        plt.plot(t_train, combined_model(t_train-1850, *popt), label="Low-frequency Model",color="C0",ls="dashed")
        plt.plot(t_train, signal_train_res, label="Residual Signal",color="C0",ls="dotted")
        plt.plot(t_test, signal_test, label="Original Signal (Testing)",color="C1",lw=2)
        plt.plot(t_test, low_freq_forecast, label="Low-freq Forecast",color="C1",ls="dashed")
        plt.plot(t_test, residual_forecast, label="Residual Forecast",color="C1",ls="dotted")
        plt.plot(t_test, forecast, label="Forecast",color="C1")
        plt.vlines(t_test[0]-.5,-150,150,color='k')
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


# wrap into xarray dataset:
fcvf = xr.Dataset(
    data_vars={
        "forecast":(("init_year","lead_time"),FORECAST),
        "verification":(("init_year","lead_time"),VERIFICATION),
    },
    coords={
        "init_year":("init_year",init_years),
        "lead_time":("lead_time",np.arange(1,step+1)),
        "valid_year":(("init_year","lead_time"),VALID_YEAR)
    }
)

# Correlation by lead time:
corr_by_lt = xr.corr(fcvf.forecast,fcvf.verification,dim="init_year")

# RMSE:
rmse_by_lt = ((fcvf.forecast - fcvf.verification)**2).mean(dim="init_year",skipna=True)**.5

# Plot scores:
f_corr,ax_corr = plt.subplots()
corr_by_lt.plot(ax=ax_corr,label="correlation")
(rmse_by_lt/fcvf.verification.std("init_year")).plot(ax=ax_corr,label="RMSE")
ax_corr.legend()

# Quick visualization:
fig,ax = plt.subplots()
fcvf.forecast.plot(x="init_year",hue="lead_time",add_legend=False,color='grey',ax=ax);
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
