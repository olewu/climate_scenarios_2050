
import xarray as xr
import pandas as pd
import numpy as np
from numpy.random import PCG64DXSM
import re
from scipy import stats
import proplot as pplt
import matplotlib.pyplot as plt
from pathlib import Path
from climate_scenarios_2050.config import proj_base, CMIP6_base, model_id_file
from climate_scenarios_2050.filters import butter_bandpass_filter, butter_lowpass_filter
from climate_scenarios_2050.utils import subsample, load_model_ensemble_NAO
from climpred import HindcastEnsemble


def pearsonr_ci(x,y,alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    r, p = stats.pearsonr(x,y)
    ac_prod = stats.pearsonr(x[1:],x[:-1])[0] * stats.pearsonr(y[1:],y[:-1])[0]
    N_eff = x.size * ((1-ac_prod)/(1+ac_prod))
    r_z = np.arctanh(r)
    se = 1/np.sqrt(N_eff-3)
    z = stats.norm.ppf(1-alpha/2)
    lo_z, hi_z = r_z-z*se, r_z+z*se
    lo, hi = np.tanh((lo_z, hi_z))
    return r, p, lo, hi


# ---------------------------
# Moving-block bootstrap (paired blocks)
# ---------------------------
def moving_block_bootstrap_pairs(x, y, L, n_boot=2000, measure = "corr"):
    """
    Paired moving-block bootstrap.
    - x, y: 1D arrays of same length N
    - L: block length (integer)
    - n_boot: number of bootstrap replicates
    Returns: array of bootstrap correlations (length n_boot)
    """
    
    N = len(x)
    # list of start indices of blocks (0..N-1) (wrap-around optional: here use non-wrap)
    starts = np.arange(0, N - L + 1)
    n_blocks_needed = int(np.ceil(N / L))
    boot_rs = np.empty(n_boot)
    # Create independent child seeds to ensure that each parallel work gets its own random state
    ss = np.random.SeedSequence(12345) # master seed
    child_seeds = ss.spawn(n_boot)
    for b in range(n_boot):
        rng = np.random.Generator(PCG64DXSM(child_seeds[b]))
        chosen = rng.integers(0, len(starts), size=n_blocks_needed)
        xs = []
        ys = []
        for c in chosen:
            s = starts[c]
            xs.append(x[s:s+L])
            ys.append(y[s:s+L])
        xs = np.concatenate(xs)[:N]
        ys = np.concatenate(ys)[:N]
        if measure == "corr":
            boot_rs[b] = stats.pearsonr(xs, ys)[0]
        elif measure == "rmse":
            boot_rs[b] = 1 - (np.nanmean((xs - ys)**2)**.5)/np.nanstd(xs)
    return boot_rs

# ---------------------------
# Stationary bootstrap (paired)
# ---------------------------
def stationary_bootstrap_pairs(x, y, L_avg, n_boot=2000, rng=None):
    """
    Paired stationary bootstrap (Politis & Romano).
    - L_avg: average block length
    """
    if rng is None:
        rng = np.random.default_rng()
    p = 1.0 / L_avg
    N = len(x)
    boot_rs = np.empty(n_boot)
    for b in range(n_boot):
        xs = np.empty(N)
        ys = np.empty(N)
        i = 0
        while i < N:
            # choose a starting index uniformly
            start = rng.integers(0, N)
            # geometric block length
            # each next sample continues block with prob (1-p)
            block_len = 0
            while i < N and rng.random() > p:  # geometric: stop with prob p
                # draw one element from start+block_len (wrap)
                idx = (start + block_len) % N
                xs[i] = x[idx]
                ys[i] = y[idx]
                i += 1
                block_len += 1
                # safety if infinite loop
                if block_len > 10*N:
                    break
            # if loop ended because rng.random() <= p, that element hasn't been added yet;
            # add at least one element per block:
            if i < N and block_len == 0:
                idx = start % N
                xs[i] = x[idx]
                ys[i] = y[idx]
                i += 1
        boot_rs[b] = stats.pearsonr(xs, ys)[0]
    return boot_rs

# ---------------------------
# Independent block bootstrap for null (destroy cross-correlation)
# ---------------------------
def independent_block_bootstrap_null(x, y, L, n_boot=2000, rng=None):
    """
    For each replicate, create x* by sampling blocks from x and y* by sampling blocks from y independently,
    then compute corr(x*, y*). This gives null distribution where marginal dependence preserved, cross-dependence destroyed.
    """
    if rng is None:
        rng = np.random.default_rng()
    N = len(x)
    starts_x = np.arange(0, N - L + 1)
    starts_y = np.arange(0, N - L + 1)
    n_blocks_needed = int(np.ceil(N / L))
    boot_rs = np.empty(n_boot)
    for b in range(n_boot):
        # sample for x*
        chosen_x = rng.integers(0, len(starts_x), size=n_blocks_needed)
        xs = []
        for c in chosen_x:
            s = starts_x[c]
            xs.append(x[s:s+L])
        xs = np.concatenate(xs)[:N]
        # sample for y* independently
        chosen_y = rng.integers(0, len(starts_y), size=n_blocks_needed)
        ys = []
        for c in chosen_y:
            s = starts_y[c]
            ys.append(y[s:s+L])
        ys = np.concatenate(ys)[:N]
        boot_rs[b] = stats.pearsonr(xs, ys)[0]
    return boot_rs

# ---------------------------
# Phase-randomized surrogates (for null)
# ---------------------------
def phase_randomized_surrogate(child_seed,series):
    """Produce a surrogate with same power spectrum: randomize Fourier phases (Keane & Potter style)."""
    rng = np.random.Generator(PCG64DXSM(child_seed))
    
    N = len(series)
    # FFT
    f = np.fft.rfft(series)
    amplitudes = np.abs(f)
    phases = np.angle(f)
    # rfft produces N//2+1 complex numbers including Nyquist
    # randomize phases for positive frequencies (except DC and Nyquist if present)
    K = len(phases)
    # indices to randomize: 1..K-2 if Nyquist exists at K-1
    # general approach: leave DC (0) and, if even N, Nyquist (last) untouched
    rand_phases = phases.copy()
    start = 1
    end = K - 1 if (N % 2 == 0) else K  # exclude Nyquist when N even
    random_vals = rng.uniform(0, 2*np.pi, size=max(0, end - start))
    rand_phases[start:end] = random_vals
    # construct new complex spectrum
    new_f = amplitudes * np.exp(1j * rand_phases)
    surrogate = np.fft.irfft(new_f, n=N)
    # adjust mean and variance to match original
    surrogate = (surrogate - surrogate.mean()) / surrogate.std() * np.std(series) + np.mean(series)
    return surrogate

def null_dist_phase_randomization(x, y, n_boot=2000, measure="corr"):
    """Create null distribution by pairing x with phase-randomized versions of y"""
    
    ss = np.random.SeedSequence(12345) # master seed
    child_seeds = ss.spawn(n_boot)
    boot_rs = np.empty(n_boot)
    for b in range(n_boot):
        ys = phase_randomized_surrogate(child_seeds[b],y)
        if measure == "corr":
            boot_rs[b] = stats.pearsonr(x, ys)[0]
        elif measure == "rmse":
            boot_rs[b] = 1 - (np.nanmean((x - ys)**2)**.5)/np.nanstd(x)
    return boot_rs

# ---------------------------
# Putting it together: CI and p-value
# ---------------------------
def bootstrap_ci_from_bootrs(boot_rs, alpha=0.05, method='percentile'):
    """Compute CI from bootstrap distribution of r (percentile by default)."""
    if method == 'percentile':
        lo = np.quantile(boot_rs, alpha/2)
        hi = np.quantile(boot_rs, 1 - alpha/2)
        return lo, hi
    else:
        raise ValueError("Only 'percentile' implemented here.")

def bootstrap_pvalue_null(observed_r, null_rs, two_sided=True):
    """Compute bootstrap p-value given null distribution of correlations (null_rs)."""
    if two_sided:
        # proportion of |null| >= |obs|
        p = np.mean(np.abs(null_rs) >= abs(observed_r))
    else:
        p = np.mean(null_rs >= observed_r)
    # if p == 0, assign 1/(n_boot+1)
    if p == 0:
        p = 1.0 / (len(null_rs) + 1)
    return p

def prep_subs_hc(ds):
    ds_cp = ds.copy()
    if "member_id" in ds.coords:
        ds_cp = ds_cp.drop("member_id")
    if "m_id" in ds.coords:
        ds_cp = ds_cp.drop("m_id")
    return ds_cp

if __name__ == "__main__":

    fpath = Path('/projects/NS9873K/www/scenarios_2050/figures')

    parameter_sets = [
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
            # {'subsampling' : "varsplit_7y", 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
            # {'subsampling' : 'Smith_incl_best', 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
            {'subsampling' : 'Smith_excl_worst', 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
            # {'subsampling' : None, 'ref_ds' : 'ERA5', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 70, 'mxlags' : 25,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : "all", 'mxlags' : 25,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 75, 'mxlags' : 25,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
            # {'subsampling' : 'varsplit_7y', 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
            # {'subsampling' : 'NAO_AR1_D2.5', 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
            # {'subsampling' : 'NAO_AR1_D2.5', 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 12, 'vf_win_len' : 12, 'mxlags' : 6,},
            # {'subsampling' : 'Smith_incl_best', 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
            # {'subsampling' : 'Smith_excl_worst', 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 20,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 12, 'vf_win_len' : 12, 'mxlags' : 6,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : 20, 'mxlags' : 15,},
            # {'subsampling' : None, 'ref_ds' : 'ERA5', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : 30, 'mxlags' : 15,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : 30, 'mxlags' : 15,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : "all", 'mxlags' : 15,},
            # {'subsampling' : "NAO_AR1_D2.5", 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : 30, 'mxlags' : 15,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 12,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp370', 'cutoff_period_short' : 7,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 6,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 12, 'vf_win_len' : 12, 'mxlags' : 6,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 30, 'vf_win_len' : 20, 'mxlags' : 15,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 12,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 25,},
            # {'subsampling' : None, 'ref_ds' : 'HadSLP', 'scen' : 'ssp245', 'cutoff_period_short' : 7,'cutoff_period_long' : 50, 'vf_win_len' : 50, 'mxlags' : 6,},
        ]

    # processing: how to filter/aggregate/average the data
    proc_type = 'running_mean'
    Nyr = 8

    # lead_times: what lead times to verify
    lead_times = np.arange(0,36,1)
    lead_times = lead_times[lead_times != 0] # zero lead time is ambiguous - remove

    linestyle = {5:'solid',15:'dashed',25:'dotted',35:'solid'}

    index_name = "NAO"
    filter_ext = "f-None"

    #----------------------------------------------------------------------------------#
    # Raw CMIP6 ensemble
    #----------------------------------------------------------------------------------#

    #-----Create a "baseline" hindcast that just takes the CMIP6 ensemble without syncing
    # reference: what observational dataset to use as a reference:

    dummy_params = parameter_sets[0]

    # load all "non-initialized" hindcasts:
    index_int_var_model = xr.open_dataarray(CMIP6_base/"internal/internal-{2}-variability_hist-{0}_{1}.nc".format(dummy_params["scen"],filter_ext,index_name))

    # process (average/filter):
    if proc_type == 'running_mean':
        index_int_var_model_proc = index_int_var_model.rolling(year=Nyr,center=True).mean('year').dropna('year',how='all').compute()
    elif proc_type == 'lowpass':
        index_int_var_model_proc = xr.apply_ufunc(
            butter_lowpass_filter,index_int_var_model,1/7,1,
            input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
            dask='allowed',vectorize=True,
        ).compute()

    # expand the array to look like a hindcast:
    index_int_var_model_proc = index_int_var_model_proc.expand_dims(dim={'init_year':len(index_int_var_model_proc.year)}).assign_coords({'init_year':('init_year',index_int_var_model_proc.year.values)})
    index_int_var_model_proc = index_int_var_model_proc.assign_coords(lead_time=index_int_var_model_proc.year - index_int_var_model_proc.init_year)

    # load forced signal from model:
    forced_signal_model = xr.open_dataarray(CMIP6_base/"forced/forced-{2}-variability_hist-{0}_{1}.nc".format(dummy_params["scen"],filter_ext,index_name))
    if proc_type == 'running_mean':
        index_forced_var_model_proc = forced_signal_model.rolling(year=Nyr,center=True).mean('year').dropna('year',how='all').compute()
    elif proc_type == 'lowpass':
        index_forced_var_model_proc = xr.apply_ufunc(
            butter_lowpass_filter,forced_signal_model,1/7,1,
            input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
            dask='allowed',vectorize=True,
        ).compute()


    # Observations:
    if dummy_params["ref_ds"] == "ERA5":
        obs_ind_dir = Path("/datalake/NS9873K/DATA/SFE/ERA5_processed/NAO_variability")
    else:
        obs_ind_dir = Path("/projects/NS9873K/owul/projects/climate_scenarios_2050/data/{:}".format(dummy_params["ref_ds"]))
    index_int_var_obs = xr.open_dataarray(obs_ind_dir/"internal-{1}-variability_{0}.nc".format(filter_ext,index_name))

    # process:
    if proc_type == 'running_mean':
        index_int_var_obs_proc = index_int_var_obs.rolling(year=Nyr).mean('year').dropna('year')
    elif proc_type == 'lowpass':
        index_int_var_obs_proc = xr.apply_ufunc(
            butter_lowpass_filter,index_int_var_obs,1/7,1,
            input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
            dask='allowed',vectorize=True,
        )

    forced_signal_obs = xr.open_dataarray(obs_ind_dir/"forced-{1}-variability_{0}.nc".format(filter_ext,index_name))
    # process:
    if proc_type == 'running_mean':
        index_forced_var_obs_proc = forced_signal_obs.rolling(year=Nyr).mean('year').dropna('year')
    elif proc_type == 'lowpass':
        index_forced_var_obs_proc = xr.apply_ufunc(
            butter_lowpass_filter,forced_signal_obs,1/7,1,
            input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
            dask='allowed',vectorize=True,
        )

    print(xr.corr(index_int_var_model_proc.isel(init_year=0).mean('member'),index_int_var_obs_proc,dim='year').values)

    # corr_raw = xr.corr(ds_hc_raw_proc.isel(init_year=0).mean('member'),vf_proc,dim='year')

    # cool plot "visual verification" of ensemble mean:
    array = [  # the "picture" (1 == subplot A, 2 == subplot B, etc.)
        [1,1,1,1],
        [1,1,1,1],
        [1,1,1,1],
        [2,2,2,2],
    ]

    # fig,axs = pplt.subplots(array,figwidth=5,figheight=4,suptitle='NAO ensemble mean hindcast')
    # axs.format(abc=True)
    # cf = axs[0].pcolormesh(index_int_var_model_proc.mean('member'),vmin=-20,vmax=20,extend='both')
    # # cf = ax.contourf(ds_hc_raw_proc.mean('member').scores)
    # axs[0].colorbar(cf, loc='r', label='')

    # axs[1].plot(index_int_var_model_proc.year,index_int_var_model_proc.isel(init_year=0).mean('member'),color='k',label='CMIP6 EM')
    # axs[1].plot(index_int_var_obs_proc.year,index_int_var_obs_proc,label=dummy_params["ref_ds"])
    # axs[1].plot(index_forced_var_obs_proc.year,index_forced_var_obs_proc,label='forced sig obs')
    # axs[1].plot(index_forced_var_model_proc.year,index_forced_var_model_proc.mean("m_id"),label='forced sig mod',ls="dotted")
    # axs[1].pcolormesh(index_int_var_obs_proc.expand_dims(dim={'dummy':2}).assign_coords({'dummy':('dummy',[75,80])}))
    # axs[0].format(
    #     xlabel='Year', ylabel='Hindcast Initialization Year',
    #     # xlim=[ds_hc_proc.year.min().values,nao_vf.year.max().values]
    #     xlim=[1879,2019], ylim=[1925,2019]
    # )

    # axs[1].format(ylim=[-80,80],ylabel='NAO index')
    # axs[1].legend(loc=3,prop={'size':6})

    # fig.savefig(fpath/'NAO_hindcast_unforced/hc_cmip6-{scen}_None_sync_raw.png'.format(**dummy_params),dpi=300,bbox_inches='tight')
    # fig.savefig(fpath/'NAO_hindcast/hc_cmip6-{scen}_None_sync_raw.png'.format(**dummy_params),dpi=300,bbox_inches='tight')

    #----------------------------------------------------------------------------------#
    # CMIP6 "initialized"/"hindcasts"
    #----------------------------------------------------------------------------------#

    # for hc_param_set in parameter_sets:
    hc_param_set = parameter_sets[0]

    # reference: what observational dataset to use as a reference:
    if hc_param_set["ref_ds"] == "ERA5":
        obs_ind_dir = Path("/datalake/NS9873K/DATA/SFE/ERA5_processed/NAO_variability")
    else:
        obs_ind_dir = Path("/projects/NS9873K/owul/projects/climate_scenarios_2050/data/{:}".format(hc_param_set["ref_ds"]))
    index_int_var_obs = xr.open_dataarray(obs_ind_dir/"internal-{1}-variability_{0}.nc".format(filter_ext,index_name))

    # process:
    if proc_type == 'running_mean':
        index_int_var_obs_proc = index_int_var_obs.rolling(year=Nyr,center=True).mean('year').dropna('year')
    elif proc_type == 'lowpass':
        index_int_var_obs_proc = xr.apply_ufunc(
            butter_lowpass_filter,index_int_var_obs,1/7,1,
            input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
            dask='allowed',vectorize=True,
        )
    # create the path the hindcast are saved to:
    hc_path = proj_base/'data/CMIP6_hindcast/hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx'.format(**hc_param_set)

    # load all hindcasts:
    hindcast_files = sorted(list(hc_path.glob('hc_*.nc')))
    ds_hc = xr.open_mfdataset(hindcast_files,preprocess=prep_subs_hc)

    # subsample if requested:
    # if hc_param_set["subsampling"] is not None:
        # ds_hc = subsample(ds_hc,subsampling=hc_param_set["subsampling"],NAO_ref=hc_param_set["ref_ds"])

    # process (average/filter):
    if proc_type == 'running_mean':
        ds_hc_proc = ds_hc['scores'].rolling(year=Nyr,center=True).mean('year').dropna('year',how='all').compute()
    elif proc_type == 'lowpass':
        ds_hc_proc = xr.apply_ufunc(
            butter_lowpass_filter,ds_hc['scores'],1/7,1,
            input_core_dims=[['year'],[],[]],output_core_dims=[['year']],
            dask='allowed',vectorize=True,
        ).compute()

    # forced signal is already removed from the hindcasts
    ds_hc_ensmean = ds_hc_proc.mean('member')


    # hindcast = HindcastEnsemble(
    #     ds_hc_proc
    # )

    # cool plot "visual verification" of ensemble mean:
    array = [  # the "picture" (1 == subplot A, 2 == subplot B, etc.)
        [1, ],
        [1, ],
        [1, ],
        [2, ],
    ]

    one_init_year = 1950

    fig,axs = pplt.subplots(array,figwidth=5,figheight=4.8,suptitle='NAO ensemble mean hindcast')
    axs.format(abc=True)
    # cf = axs[0].contourf(ds_hc_ensmean.where(ds_hc_proc.lead_time!=0),vmin=-20,vmax=20,extend='both')
    cf = axs[0].pcolormesh(ds_hc_ensmean.where(ds_hc_proc.lead_time>0),vmin=-20,vmax=20,extend='both')
    axs[0].pcolormesh(ds_hc_ensmean.where(ds_hc_proc.lead_time<0),colors="white")
    axs[0].set_facecolor("lightgrey")
    # single_hc_masked = ds_hc_ensmean.where(ds_hc_proc.lead_time!=0).sel(init_year=one_init_year)
    # single_hc = ds_hc_ensmean.sel(init_year=one_init_year)
    # cf = axs[0].pcolormesh(single_hc_masked.expand_dims(dim={'dummy':2}).assign_coords({'dummy':('dummy',[1949.5,1950.5])}),vmin=-20,vmax=20,extend='both')

    # cf = ax.contourf(ds_hc_ensmean.scores)
    axs[0].contour(ds_hc_proc.sel(year=ds_hc_proc.year.isin(index_int_var_obs_proc.year)).lead_time,levels=[ 5, 15,25,35],colors='k',linestyles='dotted',labels=True)
    # axs[0].contour(ds_hc_proc.sel(year=ds_hc_proc.year.isin(vf_proc.year)).lead_time,levels=[5],colors='k',linestyles='solid',labels=True)
    axs[0].colorbar(cf, loc='r', label='')
    # axs[0].vlines(one_init_year+np.arange(5,31,10),ds_hc_proc.init_year.min().values,1950,color='k',linestyle='dashed',linewidth=.5)
    # axs[0].fill_between([one_init_year-(hc_param_set['vf_win_len']+1),one_init_year],[1949.5,1949.5],[1950.5,1950.5],alpha=.2,color='k',zorder=100)

    axs[1].plot(index_int_var_obs_proc.year,index_int_var_obs_proc,label="HadSLP")
    # axs[1].plot(single_hc.year,single_hc,color='k')
    # lt5_hc = ds_hc_ensmean.groupby(ds_hc_ensmean.lead_time).__getitem__(5)
    # axs[1].plot(lt5_hc.year,lt5_hc,color='k')
    # axs[1].vlines(one_init_year+np.arange(5,26,10),-80,80,color='k',linestyle='dashed',linewidth=.5)
    # axs[1].fill_between([one_init_year-(hc_param_set['vf_win_len']+1),one_init_year],[-80,-80],[80,80],alpha=.2,color='k',zorder=100)
    # axs[1].text(one_init_year-hc_param_set['vf_win_len']//2,-75,s='synchronization window',va='bottom',ha='center',fontsize=8)
    # axs[1].text(one_init_year+5,80,s='LT5Y',va='bottom',ha='center',fontsize=6)
    # axs[1].text(one_init_year+15,80,s='LT15Y',va='bottom',ha='center',fontsize=6)
    # axs[1].text(one_init_year+25,80,s='LT25Y',va='bottom',ha='center',fontsize=6)
    # axs[1].text(one_init_year+1.5,55,s='"forecast"',va='top',ha='left',fontsize=8,bbox={'boxstyle':'rarrow','alpha':.7,'facecolor':'white'})
    # axs[1].arrow(one_init_year,30,10,0,head_width=10,head_length=2)
    axs[1].pcolormesh(index_int_var_obs_proc.expand_dims(dim={'dummy':2}).assign_coords({'dummy':('dummy',[75,80])}),)

    for lt in [5,15,25]:
        # group hindcasts by lead time:
        lt_group = ds_hc_ensmean.groupby(ds_hc_ensmean.lead_time).__getitem__(lt)
        lt_group = lt_group.swap_dims({"stacked_init_year_year": "year"}).drop(['stacked_init_year_year','lead_time','init_year'])
        lt_group = lt_group.assign_coords(year=pd.Index(lt_group.year.values))
        
        axs[1].plot(lt_group.year,lt_group,ls=linestyle[abs(lt)],lw=.75,color='k',label='LT{:}Y'.format(abs(lt)))


    axs[0].format(
        xlabel='Year', ylabel='Hindcast Initialization Year',
        xlim=[ds_hc_proc.init_year.min().values,index_int_var_obs_proc.year.max().values]
    )

    axs[1].format(ylim=[-80,80],ylabel='NAO index')
    axs[1].legend(loc='b',fontsize=8,ncols=4, frame=False)

    bbox = axs[0].get_position()
    width = bbox.width
    height = bbox.height
    angle = np.rad2deg(np.arctan2(height / width, 1))

    lock_year = 1985
    if hc_param_set["vf_win_len"] != "all":
        axs[0].annotate(text='', xy=(lock_year,lock_year), xytext=(lock_year-hc_param_set["vf_win_len"],lock_year), arrowprops=dict(arrowstyle='<->'))
        axs[0].text(lock_year-hc_param_set["vf_win_len"]//2,lock_year,'matching window ({:} years)'.format(hc_param_set["vf_win_len"]),ha='center',va='center',rotation=37,bbox={"fc":"lightgrey","ec":"none","pad":0})
    # fig.savefig(fpath/"illustration/hc3.png",dpi=300,bbox_inches='tight')

    # fig.savefig(fpath/'NAO_hindcast_unforced/hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx.png'.format(**hc_param_set),dpi=300,bbox_inches='tight')


    # Plot the hindcasts for a fixed lead time:
    fig_lt = pplt.figure(width=5,height=2,suptitle='NAO prediction')
    # fig_x = pplt.figure(); ax_c = fig_x.add_subplot()
    ax = fig_lt.add_subplot()
    corr = []; p_val = []; ci_lower = []; ci_upper = []
    for ii,lt in enumerate(lead_times):
        if (hc_param_set["ref_ds"] == "ERA5") and (lt > 15):
            continue
        # group hindcasts by lead time:
        lt_group = ds_hc_ensmean.groupby(ds_hc_ensmean.lead_time).__getitem__(lt)
        lt_group = lt_group.swap_dims({"stacked_init_year_year": "year"}).drop(['stacked_init_year_year','lead_time','init_year'])
        lt_group = lt_group.assign_coords(year=pd.Index(lt_group.year.values))
        
        # Correlation score of the hindcasts at lead time lt:
        year_intersect = np.intersect1d(index_int_var_obs_proc.year,lt_group.year)
        # print(lt,len(year_intersect))
        vf = index_int_var_obs_proc.sel(year=year_intersect).values
        fc = lt_group.sel(year=year_intersect).values
        cor,_ = stats.pearsonr(vf,fc)
        
        # compute p-value from phase randomization:
        null_phase_rs = null_dist_phase_randomization(vf, fc, n_boot=500)
        pval_phase = bootstrap_pvalue_null(cor, null_phase_rs)

        boot_rs = moving_block_bootstrap_pairs(vf, fc, L=max(int(np.round(len(vf)**(1/3))),Nyr), n_boot=500)
        ci_lo, ci_hi = bootstrap_ci_from_bootrs(boot_rs, alpha=0.1)

        # corr.append(xr.corr(vf_proc,lt_group,dim='year'))
        corr.append(cor); p_val.append(pval_phase); ci_lower.append(ci_lo); ci_upper.append(ci_hi)
        
        # Plot some of the lead times:
        if ii == 0:
            min_year = lt_group.year[0].values
        if lt in [5,15,25]:
            ax.plot(lt_group.year,lt_group,ls=linestyle[abs(lt)],color='k',label='{:} yr prediction'.format(abs(lt)))
            # ax_c.hist(null_phase_rs,alpha=.2)

    ax.plot(index_int_var_obs_proc.year,index_int_var_obs_proc,label='HadSLP')
    ax.format(xlim=[min_year,index_int_var_obs_proc.year.max().values],ylabel='NAO index')

    ax.legend(loc='t',prop={'size':8},ncol=4)
    # fig_lt.savefig(fpath/'NAO_hindcast_unforced/hc_cmip6-{scen}_fixedlead_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx.png'.format(**hc_param_set),dpi=300,bbox_inches='tight')

    if hc_param_set["ref_ds"] == "ERA5":
        corr_hc = xr.DataArray(
            data = corr,
            coords = {'lead_time':('lead_time',lead_times[:15])}
        )
    else:
        corr_hc = xr.DataArray(
            data = corr,
            coords = {'lead_time':('lead_time',lead_times)}
        )
    # corr_raw = xr.DataArray(
    #     data = corr_raw,
    #     coords = {'lead_time':('lead_time',lead_times)}
    # )
    # corr_hc = xr.concat(corr,dim='lead_time').assign_coords({'lead_time':('lead_time',lead_times)})
    # corr_raw = xr.concat(corr_raw,dim='lead_time').assign_coords({'lead_time':('lead_time',lead_times)})

    fig_corr = pplt.figure(suptitle='NAO Correlation Skill',figsize=[5,3])
    axc = fig_corr.add_subplot(xlabel='lead time [a]',ylabel='Correlation')
    axc.plot(corr_hc,color='C1',label='All members')
    axc.plot(corr_hc.sel(lead_time=np.array(p_val) < .1),lw=0,marker='o',color='C1',markerfacecolor="None")
    axc.plot(corr_hc.sel(lead_time=np.array(p_val) < .05),lw=0,marker='x',color='C1')
    # axc.plot(corr_hc.lead_time,p_val)
    # axc.plot(cor0r_hc_,color='C1',label='Subsampled')
    # axc.plot(corr_hc_.sel(lead_time=np.array(p_val_) < .1),lw=0,marker='o',color='C1',markerfacecolor="None")
    # axc.plot(corr_hc_.sel(lead_time=np.array(p_val_) < .05),lw=0,marker='x',color='C1')
    axc.plot([5.5,5.5],[0.57,0.79],color='k',marker='_')
    # axc.annotate(text="  DCPP (LY2-9)",xy=(5.5,0.67),ha="left",va="center")
    axc.fill_between(corr_hc.lead_time,ci_lower,ci_upper,alpha=.2,color='C1')
    # axc.fill_between(corr_hc_.lead_time,ci_lower_,ci_upper_,alpha=.2,color='C1')
    # axc.plot(corr_raw,color='C0',ls='dotted',label='raw')
    # axc.fill_between(corr_raw.lead_time,ci_lower_raw,ci_upper_raw,alpha=.2,color='C0')
    axc.hlines(0,lead_times[0],lead_times[-1],color='grey',linestyle='dashed')
    # axc.legend(loc=3,ncol=1)
    axc.format(ylim=[-.4,.9],xlim=[lead_times[0],lead_times[-1]])
    # fig_corr.savefig(fpath/'NAO_hindcast_unforced/hc_cmip6-{scen}_correlation_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx.png'.format(**hc_param_set),dpi=300,bbox_inches='tight')

    # Consider the spread of the hindcasts:
    quantiles = ds_hc_proc.quantile([.1,.25,.5,.75,.9],dim="member")
    spread = quantiles.sel(quantile=.75) - quantiles.sel(quantile=.25)
    spread_90 = quantiles.sel(quantile=.9) - quantiles.sel(quantile=.1)

    quantiles_f = index_int_var_model_proc.quantile([.1,.25,.5,.75,.9],dim="member")
    spread_f = quantiles_f.sel(quantile=.75) - quantiles_f.sel(quantile=.25)
    spread_90_f = quantiles_f.sel(quantile=.9) - quantiles_f.sel(quantile=.1)

    f,ax = pplt.subplots()
    spread.groupby("lead_time").mean().plot(ax=ax)
    spread_f.groupby("lead_time").mean().plot(ax=ax)

    fig,axs = pplt.subplots(array,figwidth=5,figheight=4.8,suptitle='NAO ensemble mean hindcast')
    axs.format(abc=True)
    # cf = axs[0].contourf(ds_hc_ensmean.where(ds_hc_proc.lead_time!=0),vmin=-20,vmax=20,extend='both')
    cf = axs[0].pcolormesh(spread.where(spread.lead_time!=0),vmin=30,vmax=40,extend='both')
    axs[0].set_facecolor("lightgrey")
    # single_hc_masked = ds_hc_ensmean.where(ds_hc_proc.lead_time!=0).sel(init_year=one_init_year)
    # single_hc = ds_hc_ensmean.sel(init_year=one_init_year)
    # cf = axs[0].pcolormesh(single_hc_masked.expand_dims(dim={'dummy':2}).assign_coords({'dummy':('dummy',[1949.5,1950.5])}),vmin=-20,vmax=20,extend='both')

    # cf = ax.contourf(ds_hc_ensmean.scores)
    axs[0].contour(spread.sel(year=spread.year.isin(index_int_var_obs_proc.year)).lead_time,levels=[-30,-20,-10, 10,20,30],colors='k',linestyles='dotted',labels=True)
    # axs[0].contour(ds_hc_proc.sel(year=ds_hc_proc.year.isin(vf_proc.year)).lead_time,levels=[5],colors='k',linestyles='solid',labels=True)
    axs[0].colorbar(cf, loc='r', label='')
    # axs[0].vlines(one_init_year+np.arange(5,31,10),ds_hc_proc.init_year.min().values,1950,color='k',linestyle='dashed',linewidth=.5)
    # axs[0].fill_between([one_init_year-(hc_param_set['vf_win_len']+1),one_init_year],[1949.5,1949.5],[1950.5,1950.5],alpha=.2,color='k',zorder=100)

    axs[1].plot(index_int_var_obs_proc.year,index_int_var_obs_proc,label="HadSLP")
    # axs[1].plot(single_hc.year,single_hc,color='k')
    # lt5_hc = ds_hc_ensmean.groupby(ds_hc_ensmean.lead_time).__getitem__(5)
    # axs[1].plot(lt5_hc.year,lt5_hc,color='k')
    # axs[1].vlines(one_init_year+np.arange(5,26,10),-80,80,color='k',linestyle='dashed',linewidth=.5)
    # axs[1].fill_between([one_init_year-(hc_param_set['vf_win_len']+1),one_init_year],[-80,-80],[80,80],alpha=.2,color='k',zorder=100)
    # axs[1].text(one_init_year-hc_param_set['vf_win_len']//2,-75,s='synchronization window',va='bottom',ha='center',fontsize=8)
    # axs[1].text(one_init_year+5,80,s='LT5Y',va='bottom',ha='center',fontsize=6)
    # axs[1].text(one_init_year+15,80,s='LT15Y',va='bottom',ha='center',fontsize=6)
    # axs[1].text(one_init_year+25,80,s='LT25Y',va='bottom',ha='center',fontsize=6)
    # axs[1].text(one_init_year+1.5,55,s='"forecast"',va='top',ha='left',fontsize=8,bbox={'boxstyle':'rarrow','alpha':.7,'facecolor':'white'})
    # axs[1].arrow(one_init_year,30,10,0,head_width=10,head_length=2)
    axs[1].pcolormesh(index_int_var_obs_proc.expand_dims(dim={'dummy':2}).assign_coords({'dummy':('dummy',[75,80])}),)

    for ii,lt in enumerate(lead_times):
        # group hindcasts by lead time:
        lt_group = ds_hc_ensmean.groupby(ds_hc_ensmean.lead_time).__getitem__(lt)
        lt_group = lt_group.swap_dims({"stacked_init_year_year": "year"}).drop(['stacked_init_year_year','lead_time','init_year'])
        lt_group = lt_group.assign_coords(year=pd.Index(lt_group.year.values))
        
        # Plot some of the lead times:
        if ii == 0:
            min_year = lt_group.year[0].values
        if lt in [5,15,25]:
            axs[1].plot(lt_group.year,lt_group,ls=linestyle[abs(lt)],lw=.75,color='k',label='LT{:}Y'.format(abs(lt)))


    axs[0].format(
        xlabel='Year', ylabel='Hindcast Initialization Year',
        xlim=[ds_hc_proc.year.min().values,index_int_var_obs_proc.year.max().values]
    )

    axs[1].format(ylim=[-80,80],ylabel='NAO index')
    axs[1].legend(loc='b',fontsize=8,ncols=4, frame=False)

    bbox = axs[0].get_position()
    width = bbox.width
    height = bbox.height
    angle = np.rad2deg(np.arctan2(height / width, 1))

    axs[0].annotate(text='', xy=(one_init_year,one_init_year), xytext=(one_init_year-hc_param_set["vf_win_len"],one_init_year), arrowprops=dict(arrowstyle='<->'))
    axs[0].text(one_init_year-hc_param_set["vf_win_len"]//2,one_init_year,'synchronization window ({:} years)'.format(hc_param_set["vf_win_len"]),ha='center',va='center',rotation=37,bbox={"fc":"lightgrey","ec":"none","pad":0})
    # fig.savefig(fpath/"illustration/hc3.png",dpi=300,bbox_inches='tight')

    # fig.savefig(fpath/'NAO_hindcast_unforced/hc_cmip6-{scen}_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx-spread.png'.format(**hc_param_set),dpi=300,bbox_inches='tight')

    # re-arrange the hindcasts to have lead time as a dimension instead of valid year:
    ds_hc_mtrx = []; vf_hc_like = []; ds_fullens_mtrx = []
    for iyear in ds_hc_proc.init_year.values:
        single_init = ds_hc_proc.sel(init_year=iyear)
        single_init_shifted = single_init.sel(year=single_init.lead_time > 0).swap_dims({"year":"lead_time"})
        single_init_incl = single_init_shifted.sel(lead_time=slice(35))
        ds_hc_mtrx.append(single_init_incl)
        
        vf_matched = index_int_var_obs_proc.reindex(year=single_init_incl.year.values).assign_coords({"lead_time":("year",single_init_incl.lead_time.values)}).swap_dims({"year":"lead_time"})
        vf_hc_like.append(vf_matched)

        single_init_fullens = index_int_var_model_proc.sel(init_year=iyear)
        single_init_fullens_shifted = single_init_fullens.sel(year=single_init_fullens.lead_time > 0).swap_dims({"year":"lead_time"})
        single_init_fullens_incl = single_init_fullens_shifted.sel(lead_time=slice(35))
        ds_fullens_mtrx.append(single_init_fullens_incl)
        
    ds_hc_mtrx = xr.concat(ds_hc_mtrx,dim="init_year")
    ds_fullens_mtrx = xr.concat(ds_fullens_mtrx,dim="init_year")
    # verification
    vf_hc_like = xr.concat(vf_hc_like,dim=ds_hc_mtrx.init_year).drop("mode")

    # bias correct (not properly leave-out!)
    hc_bias = ds_hc_mtrx.mean(("init_year","member")) - vf_hc_like.mean("init_year")
    fullens_bias = ds_hc_mtrx.mean(("init_year","member")) - vf_hc_like.mean("init_year")
    # ds_hc_mtrx -= hc_bias
    # ds_fullens_mtrx -= fullens_bias

    # RMSE:
    rmse_hc = (((ds_hc_mtrx.mean("member") - vf_hc_like)**2).mean("init_year")**.5)
    rmse_ref = (((ds_fullens_mtrx.mean("member") - vf_hc_like)**2).mean("init_year")**.5)
    rmsess = 1 - rmse_hc/rmse_ref
    rmsess.plot()

    # Evaluate the ensemble with a simple Brier Score (NAO above or below 0)
    thresh_vf = vf_hc_like.quantile(.5,dim="init_year") # 0
    thresh_hc = ds_hc_mtrx.quantile(.5,dim="init_year") # index_int_var_model_proc.quantile(.5)
    thresh_mod = ds_fullens_mtrx.quantile(.5,dim="init_year") # index_int_var_model_proc.quantile(.5)

    prob_vals = np.array([0,.1,.2,.3,.4,.5,.6,.7,.8,.9,1])
    pos_mod = (ds_hc_mtrx > thresh_hc).mean("member").round(1)
    pos_fullens = (ds_fullens_mtrx > thresh_mod).mean("member").round(1)
    pos_vf = vf_hc_like > thresh_vf
    BS = ((pos_mod - pos_vf)**2).mean("init_year")
    BSref = ((pos_fullens - pos_vf)**2).mean("init_year")
    BSS = 1 - BS/BSref
    BSS.plot()

    # Reliability Diagram (0 thresh, one lead year):
    for LT in [5,10,15,20]:
        pos_mod_lt = pos_fullens.sel(lead_time=LT)
        pos_vf_lt = pos_vf.sel(lead_time=LT)
        obs_freq = []; fc_freq = []
        for prob in prob_vals:
            fc_freq.append((pos_mod_lt==prob).sum().values*10)
            cat_masked = pos_vf_lt.where(pos_mod_lt==prob)
            obs_freq.append(cat_masked.sum().values/np.isfinite(cat_masked).sum().values*100)

        plt.figure()
        plt.scatter(prob_vals*100,obs_freq,s=fc_freq)
        plt.plot([0,100],[0,100],'k')
        plt.title(f"Reliability {LT}Y prediction")

    # Evaluate three category ranked probability score for terciles
    # derive terciles from obs:



    # TODO:
    # evaluate spread --> RPSS?
    # RMSE? then, need to standardize somehow. difficult to say what stdzation would make sense...
    # Random Walk test? DelSole & Tippett (2015) 


    # fig_corr = pplt.figure(suptitle='NAO Correlation Skill',fontsize=13,figsize=[3.5,2.4])
    # axc = fig_corr.add_subplot(xlabel='lead time [a]',ylabel='Correlation',fontsize=12)
    # axc.plot(corr_hc,color='C1',label='All members')
    # axc.plot(corr_hc.sel(lead_time=np.array(p_val) < .1),lw=0,marker='o',color='C1',markerfacecolor="None")
    # # axc.plot(corr_hc.sel(lead_time=np.array(p_val) < .05),lw=0,marker='x',color='C1')
    # # axc.plot(corr_hc_,color='C1',label='Subsampled')
    # # axc.plot(corr_hc_.sel(lead_time=np.array(p_val_) < .1),lw=0,marker='o',color='C1',markerfacecolor="None")
    # # axc.plot(corr_hc_.sel(lead_time=np.array(p_val_) < .05),lw=0,marker='x',color='C1')
    # # axc.plot([5.5,5.5],[0.57,0.79],color='k',marker='_')
    # # axc.annotate(text="  DCPP (LY2-9)",xy=(5.5,0.67),ha="left",va="center")
    # axc.fill_between(corr_hc.lead_time,ci_lower,ci_upper,alpha=.2,color='C1')
    # # axc.fill_between(corr_hc_.lead_time,ci_lower_,ci_upper_,alpha=.2,color='C1')
    # # axc.plot(corr_raw,color='C0',ls='dotted',label='raw')
    # # axc.fill_between(corr_raw.lead_time,ci_lower_raw,ci_upper_raw,alpha=.2,color='C0')
    # axc.hlines(0,lead_times[0],lead_times[-1],color='grey',linestyle='dashed')
    # # axc.legend(loc=3,ncol=1)
    # axc.format(ylim=[-.4,.8],xlim=[lead_times[0],25])
    # fig_corr.savefig(fpath/'pres_hc_cmip6-{scen}_correlation_{subsampling}_sync_{ref_ds}_bpy{cutoff_period_short}-{cutoff_period_long}_{vf_win_len}yvf_{mxlags}ymx.png'.format(**hc_param_set),dpi=300,bbox_inches='tight')
