# Compare the model NAO simulations to observations
# Method from DelSole & Tippett (2020) Comparing Climate Time Series - Part I: Univariate Test
# Tests whether AR(p)-model parameters for two time series are significantly different

from statsmodels.tsa.ar_model import AutoReg
# from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from climate_scenarios_2050.generate_NAO_hindcast import load_model_ensemble_NAO
from climate_scenarios_2050.config import *
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import itertools
import matplotlib.colors as mplcolors
from scipy.stats import f
import re
import pickle

subsampling_path = proj_base/'data/subsampling'


with open(model_id_file,'rb') as file:
    model_id = pickle.load(file)

model_id_inv = {val:key for key,val in model_id.items()}

def extract_x(val):
    match = re.match(r"m(\d+)r\d+i\d+p\d+f\d+", val)
    return int(match.group(1)) if match else None


n_bs = 1000

ref_ds = 'HadSLP'

NAO_int = xr.open_dataarray(CMIP6_base/"internal/internal-NAO-variability-hist-{0}_f-None.nc".format("ssp370"))
NAO_forced  = xr.open_dataarray(CMIP6_base/"forced/forced-NAO-variability_hist-{0}_f-None.nc".format("ssp370"))
NAO_sim = NAO_int# + NAO_forced.sel(m_id=NAO_int.m_id)

# NAO_sim = load_model_ensemble_NAO(None,'ssp245',ref_ds)
# years_full = np.arange(NAO_sim.year.min(), NAO_sim.year.max() + 1)
# NAO_sim = NAO_sim.reindex(year=years_full).interpolate_na(dim='year', method='linear')

obs_ind_dir = Path("/projects/NS9873K/owul/projects/climate_scenarios_2050/data/HadSLP")
nao_vf_int = xr.open_dataarray(obs_ind_dir/"internal-NAO-variability_f-None.nc")
nao_vf_forced = xr.open_dataarray(obs_ind_dir/"forced-NAO-variability_f-None.nc")
nao_vf_full = nao_vf_int# + nao_vf_forced

# nao_vf_full = xr.open_dataset(proj_base/'data/{0:s}/{0:s}_psl_EOF_full_index_DJFM_1850-2019_1.5deg.nc'.format(ref_ds))
# nao_vf_full = nao_vf_full.sel(mode=1).squeeze().scores

# em = NAO_sim.mean('member')

# remove forced signal TODO: need to change this!
# x = (nao_vf_full - em.scores).values
x = nao_vf_full.values

p = 1

resx = AutoReg(x, lags = p, trend = 'ct').fit()
phi_x = resx.params[2:] # note that constant is on 0, and trend on 1 if both are selected!
# residual variance:
sigmax = resx.resid.var()
# degrees of freedom:
nux = len(x) - 2*p - 1
Z_x = np.column_stack([x[i:len(x) - p + i] for i in range(p)])
Z_x = Z_x - Z_x.mean(0)

D_phisig_models = []

for mem in NAO_sim.member:

    # take model simulation over the same period as observations:
    y = NAO_sim.sel(member=mem).sel(year=nao_vf_full.year).values

    resy = AutoReg(y, lags = p, trend = 'ct').fit()

    phi_y = resy.params[2:] # note that constant is on 0, and trend on 1 if both are selected!

    # residual variance:
    sigmay = resy.resid.var()
    # degrees of freedom:
    nuy = len(y) - 2*p - 1

    F_sig = sigmax/sigmay # ~F_nux,nuy

    sigma = (nux*sigmax + nuy*sigmay)/(nux+nuy)

    Z_y = np.column_stack([y[i:len(y) - p + i] for i in range(p)])
    Z_y = Z_y - Z_y.mean(0)

    sigma_HM = np.linalg.inv(np.linalg.inv(Z_x.T@Z_x) + np.linalg.inv(Z_y.T@Z_y))

    F_phisig = ((phi_x - phi_y).T@sigma_HM@(phi_x - phi_y))/p/sigma # ~F_p,(nux+nuy)

    D_sig = nux * np.log(nux + nuy/F_sig) + nuy * np.log(nux*F_sig + nuy) - (nux+nuy)*np.log(nux+nuy)

    D_phicond = (nux+nuy) * np.log(1 + (p*F_phisig/(nux+nuy)))

    D_phisig = D_sig + D_phicond

    D_phisig_models.append(D_phisig)

    # print(D_phisig)

D_phisig_models = xr.DataArray(
    np.array(D_phisig_models),
    coords={'member':NAO_sim.member.values},
    dims=('member',)
)

D_phisig_models = D_phisig_models.assign_coords(model=("member", [extract_x(val) for val in D_phisig_models["member"].values]))

D_phisig_models_sorted = D_phisig_models.sortby(D_phisig_models)

# bootstrap D (depends on degrees of freedom and p only, so is independent of each individual simulation!):
F_sig_rand = f.rvs(nux,nuy,size=n_bs)
F_phisig_rand = f.rvs(p,nux+nuy,size=n_bs)

D_sig_rand = nux * np.log(nux + nuy/F_sig_rand) + nuy * np.log(nux*F_sig_rand + nuy) - (nux+nuy)*np.log(nux+nuy)
D_phicond_rand = (nux+nuy) * np.log(1 + (p*F_phisig_rand/(nux+nuy)))

D_phisig_rand = np.tile(D_sig_rand,[n_bs,1]).T.flatten() + np.tile(D_phicond_rand,[n_bs,1]).flatten()

# 5% confidence:
sig_lev = 2.5

D_phisig_percentiles = np.percentile(D_phisig_rand,100-sig_lev)

# Visualize
# Group by model
# Step 1: Compute the average value of each X group
group_means = D_phisig_models.groupby("model").mean()

# Step 2: Sort X values by group mean
sorted_model = group_means.sortby(group_means).model.values

# Step 3: Within each X, sort members by their mean over time
sorted_members = []
for xval in sorted_model:
    group = D_phisig_models.sel(member=D_phisig_models.model == xval)
    ordered_members = np.sort(group.member.values)
    sorted_members.extend(ordered_members)

# Step 4: Reindex the original DataArray
sorted_data = D_phisig_models.sel(member=sorted_members)

group_map = {mem:extract_x(mem) for mem in sorted_data.member.values}

df = sorted_data.to_series().reset_index()
df['model'] = df['member'].map(group_map)

color_cycle = itertools.cycle(mplcolors.TABLEAU_COLORS)

fig, ax = plt.subplots(figsize=(8, 4))
plt.axhline(D_phisig_percentiles,color='k',ls='dotted');
# loop over all groups and sort the bars by size (D)
st = 0; grp_pos = []; grp_label = []; sel_grp = []
for model,color in zip(df.model.unique(),color_cycle):
    df_plot = df[df.model == model][0]
    ax.bar(np.arange(st,st+len(df_plot)),np.sort(df_plot.values),color=color)
    ax.hlines(df_plot.mean(),st,st+len(df_plot),color=color,linestyle='dashed')
    grp_pos.append(np.arange(st,st+len(df_plot)).mean())
    grp_label.append(model_id_inv[f"m{model}"])
    st += len(df_plot) + 1
    if df_plot.mean() < D_phisig_percentiles:
        sel_grp.append(model_id_inv[f"m{model}"])

# Set custom x-ticks: show group names centered under their group
ax.set_xticks(grp_pos);
ax.set_xticklabels(grp_label,rotation=75,fontsize=5);
ax.set_xlim([0,st])

plt.xlabel('Model')
plt.ylabel('Distance')
plt.title('')
plt.tight_layout()

fig.savefig(f"/projects/NS9873K/www/scenarios_2050/figures/NAO_diff_AR/NAO_AR{p}_D{sig_lev}.png",dpi=300,bbox_inches="tight")

model_sel_file = subsampling_path/f'NAO_AR{p}_D{sig_lev}_subsampling.pkl'
with open(model_sel_file, "wb") as fp:
    model_selection = pickle.dump(sel_grp,fp)

# plt.bar(x=sorted_data.member,height=sorted_data);

# TODO
# generate a list of model simulations that pass the test of being indistinguishable from the observations

# if D_phisig_percentiles[0] < D_phisig < D_phisig_percentiles[-1]:
#     print('spectra of the two simulations are indistingushable')
# else:
#     print('significant differences between spectra of the two simulations')


# TODO: develop this to form a criterion for subsampling models
# questions:
# 1) what to do with ensemble members from the same model?