import cdsapi
import subprocess as sbp
from pathlib import Path
import re

dataset = "reanalysis-era5-single-levels-monthly-means"

dl_file = Path("/datalake/NS9873K/DATA/SFE/ERA5/res_monthly_1/era5_monthly_1940-1959.grb")
request = {
    "product_type": ["monthly_averaged_reanalysis"],
    "variable": [
        "2m_temperature",
        "total_precipitation",
        "10m_wind_speed"
    ],
    "year": [
        "1940", "1941", "1942",
        "1943", "1944", "1945",
        "1946", "1947", "1948",
        "1949", "1950", "1951",
        "1952", "1953", "1954",
        "1955", "1956", "1957",
        "1958", "1959"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "time": ["00:00"],
    "data_format": "grib",
    "download_format": "unarchived",
    "area": [90, 0, -90, 359.5],
    "grid": "0.5/0.5"
}

client = cdsapi.Client()
client.retrieve(dataset, request,f"{dl_file}")

# split file by variable:
splt_vars = dl_file.parent/dl_file.name.replace(".grb","_[shortName].grb")
cp = sbp.run(
    [
        "grib_copy",
        f"{dl_file}",
        f"{splt_vars}"
    ]
)

if cp.returncode == 0:
    dl_file.unlink()

shortn_to_name = {
    "2t":"2m_temperature",
    "tp":"total_precipitation",
    "10si":"10m_wind_speed",
}

files_all_vars = sorted(list(splt_vars.parent.glob("*1959_*.grb")))

for file_var in files_all_vars:
    
    var_code = [shortn_to_name[shn] for shn in shortn_to_name.keys() if shn in file_var.stem][0]

    # split by year & month:
    splt_yemo = file_var.parent/f"{var_code}/{var_code}_[year]_[month].grb"
    splt_ym_ret = sbp.run(
        [
             "grib_copy",
            f"{file_var}",
            f"{splt_yemo}"
        ]
    )

    if splt_ym_ret.returncode == 0:
        file_var.unlink()

    all_grb_var = sorted(list(splt_yemo.parent.glob("*.grb")))

    for grb_yrmo in all_grb_var:
        # relabel month (add leading 0):
        single_digit_month = re.search(r"_(\d{1}).grb",grb_yrmo.name)
        if single_digit_month is not None:
            find_ext = "_" + single_digit_month.group(1) + "."
            repl_ext = "_" + "{:}".format(single_digit_month.group(1).zfill(2)) + "."
            nc_yrmo = grb_yrmo.parent/grb_yrmo.name.replace(find_ext,repl_ext).replace(".grb",".nc")
        else:
            nc_yrmo = grb_yrmo.parent/grb_yrmo.name.replace(".grb",".nc")

        # convert to netcdf
        conv = sbp.run(
            [
                "grib_to_netcdf",
                "-o",
                f"{nc_yrmo}",
                f"{grb_yrmo}"
            ]
        )

        if conv.returncode == 0:
            grb_yrmo.unlink()
