import pickle
from pathlib import Path
import os

base_path = Path('/projects/NS9873K')
proj_base = Path(os.path.dirname(os.path.realpath(__file__))).parents[1]

stations = {
    # specify name as key and [lon,lat] coordinates
    'UtsiraNord':{
        'name': 'Utsira Nord',
        'shortKey': 'UN',
        'lon':4.54,
        'lat':59.25,
    },
    'CelticSea':{
        'name': 'Celtic Sea',
        'shortKey': 'CS',
        'lon':-6.24,
        'lat':51.75,
    },
    'NorfolkBoreas':{
        'name': 'Norfolk Boreas',
        'shortKey': 'NB',
        'lon':2.82,
        'lat':53.08,
    },
    'NorthernSpain':{
        'name': 'Northern Spain',
        'shortKey': 'NS',
        'lon':-8.63,
        'lat':44.03,
    },
    'Smola':{
        'name': 'Smøla',
        'shortKey': 'SM',
        'lon':6.45,
        'lat':63.47,
    },
    'NISA':{
        'name': 'NISA',
        'shortKey': 'NISA',
        'lon':-5.86,
        'lat':53.70,
    },
    'SorligeNordsjo2':{
        'name': 'Sørlige Nordsjø 2',
        'shortKey': 'SN2',
        'lon':5.04,
        'lat':56.83,
    },
}

example_grid_file = '/projects/NS9873K/DATA/SFE/ERA5/res_6hrly_1/2m_temperature/2m_temperature_2000_01.nc'

NAO_box = {
    'south':{
        'lon':slice(-28,-20),
        'longitude':slice(332,340),
        'lat':slice(36,40),
        'latitude':slice(40,36),
        },
    'north':{
        'lon':slice(-25,-16),
        'longitude':slice(335,344),
        'lat':slice(63,70),
        'latitude':slice(70,63),
        }
} # Azores (28–20° W, 36–40°N) and Iceland (25–16°W, 63–70°N), Smith et al. (2019)

# NAO domain for EOF-based definition of NAO
NAO_domain = {
    'winter':{ # following Hurrell & Deser (2009)
        'longitude':slice(-90,40),
        'latitude':slice(20,80),
    },
    'summer':{ # following Folland et al. (2009)
        'longitude':slice(-70,50),
        'latitude':slice(25,70),
    },
}

AO_domain = {
    'longitude':slice(None),
    'latitude':slice(20,89.9),
}

# mapping ERA5 names to DCPP/CMIP6:
var_name_map = {
    "cf_to_cmip":{
        "t2m":"tas",
        "tp":"pr",
        "si10":"sfcWind",
        "ssr":"rsds",
        "msl":"psl",
    },
    "long_to_cf":{
        "2m_temperature":"t2m",
        "total_precipitation":"tp",
        "10m_wind_speed":"si10",
        "surface_net_solar_radiation":"ssr",
        "mean_sea_level_pressure":"msl",
    },
    "long_to_cmip":{
        "2m_temperature":"tas",
        "total_precipitation":"pr",
        "10m_wind_speed":"sfcWind",
        "surface_net_solar_radiation":"rsds",
        "mean_sea_level_pressure":"psl",
    },
    "cmip_to_cf":{
        "tas":"t2m",
        "pr":"tp",
        "sfcWind":"si10",
        "rsds":"ssr",
        "psl":"msl",
    },
    "cf_to_long" :{
        "t2m":"2m_temperature",
        "tp":"total_precipitation",
        "si10":"10m_wind_speed",
        "ssr":"surface_net_solar_radiation",
        "msl":"mean_sea_level_pressure",
    },
    "cds_to_cmip":{
        "sea_level_pressure":"psl",
        "precipitation":"pr",
        "near_surface_air_temperature":"tas",
    }
}

unit_conversion = {
    'ERA5':{
        "t2m":1,
        "tp":1000, # to get from m to mm
        "si10":1,
        "ssr":1/86400, # to get from J/s to W/m^2
        "msl":1/100, # to get from Pa to hPa
    },
    'CMIP':{
        "t2m":1,
        "tp":86400, # to get from kg/m2/s to mm
        "si10":1,
        "ssr":1,
        "msl":1/100, # to get from Pa to hPa
    }
}

units = {
    "t2m":"˚C",
    "tp":"mm",
    "si10":"m/s",
    "ssr":"W/m^2",
    "msl":"hPa",
}

data_paths = {
    'verification':{
        'ERA5':base_path/'DATA/SFE/ERA5/res_monthly_1',
        # others?
    },
    'hindcast':{
        'NorCPM1':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/NorCPM1/'),
        'EC-Earth3':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/EC-Earth3/'),
        'HadGEM3-GC31-MM':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/HadGEM3-GC31-MM/'),
        'CMCC-CM2-SR5':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/CMCC-CM2-SR5/'),
        'MPI-ESM1-2-HR':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/MPI-ESM1-2-HR/'),
        'CanESM5':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/CanESM5/'),
        'MPI-ESM1-2-LR':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/MPI-ESM1-2-LR/'),
        'CNRM-ESM2-1':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/CNRM-ESM2-1/'),
        'MIROC6':Path('/datalake/NS9873K/DATA/DCPP/dcppA-hindcast/MIROC6/'),
    },
    'forecast':{
        'NorCPM1':Path('/projects/NS9034K/CMIP6/.cmorout/NorCPM1/dcppB-forecast'),
        # others?
    },
    'processed':base_path/'owul/data/statkraft_pilot4/decadal_predictions',
    'figures':base_path/'owul/figures/decadal_predictions',
    'figures_online':base_path/'www/decadal'
}
model_id_file = proj_base/'src/climate_scenarios_2050/model_id.pkl'

def model_mapping(base_dir,model_id_file):

    # load mapping:
    if model_id_file.exists():
        with open(model_id_file,'rb') as f:
            model_id = pickle.load(f)
    else:
        model_id = {}

    # find the existing models:
    models = {model.name:f'm{ii}' for (ii,model) in enumerate(sorted(list((base_dir/'data/CMIP6/historical').glob('*')))) if not model.name in ['OLD','url_repo']}
    # Add only new subdirectories
    next_id = max(model_id.values(), default=-1) + 1
    for subdir in models.values:
        if subdir not in model_id:
            model_id[subdir] = next_id
            next_id += 1

    
    # save mapping:
    with open(model_id_file,'wb') as f:
        pickle.dump(model_id,f)
if __name__ == '__main__':
    
    model_id_file = proj_base/'src/climate_scenarios_2050/model_id.pkl'

    model_mapping(proj_base,model_id_file)