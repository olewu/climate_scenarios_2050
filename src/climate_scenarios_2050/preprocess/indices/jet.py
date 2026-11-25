"""
Eddy-driven jet indices after Woollings et al. (2010), adapted to seasonal means: the zonal wind (850 hPa)
is averaged over the longitudes of `predictor.domain` (0-60W), and the latitude of the maximum of this
profile within the domain's latitudes (15-75N) is the jet latitude, the maximum itself the jet speed.
Woollings et al. low-pass filter daily data (and average 925-700 hPa); here the profile of the seasonal
mean field is used, on the common grid (observations and models alike).

Below-ground points: 850 hPa lies below the surface over Greenland and the Atlas. About half of the
models leave these points missing (INM writes 0, read as missing), ERA5 and 20CR extrapolate. A fixed
mask (`predictor.jet.mask`, file `configs/masks/<name>.nc`, built with `build_mask` from the CMIP6 stores)
removes these points from every dataset, so all profiles average over the same longitudes. A latitude
needs `min_valid` (default 0.8) of its unmasked longitudes (points missing in single models or years); a
latitude without enough data is left out of the maximum search.

With `refine: parabola` (default) a parabola through the maximum and its two neighbours gives a
continuous latitude and speed (on a 1.5 deg grid the plain maximum takes only ~15 distinct latitudes
in 250 years); at the edge of the latitude range or next to a missing latitude the grid value is kept.

Like the other indices, an anomaly index: each dataset's (model's) own normal-period mean of the raw
latitude/speed is removed, and the anomalies are standardized with the reference dataset's normal-period
standard deviation. The raw normal-period mean is kept (`raw_climatology`, degrees north or m/s) to
compare the models' jet with the observed one.

Building the mask (written to a new file; the file's checksum enters the index hashes):
    pixi run python -m climate_scenarios_2050.preprocess.indices.jet <out.nc> <store.zarr> ...
"""

import sys
from pathlib import Path

import numpy as np
import xarray as xr

from climate_scenarios_2050.core.config import CONFIG_DIR

QUANTITIES = {"latitude": "degrees_north", "speed": "m s-1"}
REFINE = ("parabola", "none")
MASK_DIR = CONFIG_DIR / "masks"


#---------------options and mask--------------#
def options(pred: dict) -> dict:
    """`predictor.jet` with defaults: quantity, refine, mask (DataArray or None), min_valid"""
    opts = pred.get("jet") or {}
    return {
        "quantity": opts.get("quantity", "latitude"),
        "refine": opts.get("refine", "parabola"),
        "mask": load_mask(opts["mask"]) if opts.get("mask") else None,
        "min_valid": float(opts.get("min_valid", 0.8)),
    }


def mask_file(name: str) -> Path:
    return MASK_DIR / f"{name}.nc"


def load_mask(name: str) -> xr.DataArray:
    """True where a point is excluded"""
    with xr.open_dataset(mask_file(name)) as ds:
        return ds["mask"].astype(bool).load()


def _on_grid(mask: xr.DataArray, field: xr.DataArray) -> xr.DataArray:
    m = mask.astype(float).reindex(lat=field["lat"], lon=field["lon"], method="nearest", tolerance=1e-3)
    if bool(m.isnull().any()):
        raise ValueError("predictor.jet.mask does not cover the jet domain")
    return m.astype(bool)


#---------------index--------------#
def profile(field: xr.DataArray, mask=None, min_valid: float = 0.8) -> xr.DataArray:
    """
    mean over the unmasked longitudes (equally spaced); NaN at a latitude where fewer than `min_valid`
    of them have data
    """
    if mask is None:
        possible = field.sizes["lon"]
    else:
        m = _on_grid(mask, field)
        field, possible = field.where(~m), (~m).sum("lon")
    enough = (field.notnull().sum("lon") >= min_valid * possible) & (possible > 0)
    return field.mean("lon", skipna=True).where(enough)


def jet_latitude_speed(prof: xr.DataArray, refine: str = "parabola"):
    """latitude and value of the maximum of a (..., lat) profile over its valid latitudes; NaN without any"""
    prof = prof.transpose(..., "lat")
    lat = prof["lat"].values.astype(float)
    v = prof.values.astype(float)
    flat = v.reshape(-1, lat.size)
    finite = np.isfinite(flat)
    valid = finite.any(axis=1)
    j = np.argmax(np.where(finite, flat, -np.inf), axis=1)
    rows = np.arange(flat.shape[0])
    out_lat, out_speed = lat[j].copy(), flat[rows, j].copy()
    if refine == "parabola" and lat.size >= 3:
        inner = valid & (j > 0) & (j < lat.size - 1)
        jj = np.clip(j, 1, lat.size - 2)
        inner &= finite[rows, jj - 1] & finite[rows, jj + 1]
        r, k = rows[inner], j[inner]
        y0, y1, y2 = flat[r, k - 1], flat[r, k], flat[r, k + 1]
        curv = y0 - 2 * y1 + y2
        ok = curv < 0
        d = np.where(ok, 0.5 * (y0 - y2) / np.where(ok, curv, 1.0), 0.0)
        half = (lat[k + 1] - lat[k - 1]) / 2
        out_lat[r] = lat[k] + d * half
        out_speed[r] = y1 - 0.25 * (y0 - y2) * d
    out_lat[~valid], out_speed[~valid] = np.nan, np.nan
    shape, dims = v.shape[:-1], prof.dims[:-1]
    coords = {name: c for name, c in prof.coords.items() if "lat" not in c.dims}
    as_da = lambda x: xr.DataArray(x.reshape(shape), dims=dims, coords=coords)
    return as_da(out_lat), as_da(out_speed)


def raw_index(field: xr.DataArray, quantity: str, refine: str = "parabola", mask=None, min_valid: float = 0.8) -> xr.DataArray:
    """jet latitude or speed per year (and member) of a field cut to the jet domain"""
    lat, speed = jet_latitude_speed(profile(field, mask, min_valid), refine)
    out = lat if quantity == "latitude" else speed
    out.attrs = {"units": QUANTITIES[quantity]}
    return out


def normal_period_mean(raw: xr.DataArray, normal_period) -> float:
    """mean over the normal period; for an ensemble over the members that cover it fully (else all member-years)"""
    sub = raw.sel(year=slice(*normal_period))
    if "member" in sub.dims:
        full = sub.notnull().all("year")
        if bool(full.any()):
            sub = sub.sel(member=full.values)
    return float(sub.mean())


def reference_stats(field: xr.DataArray, quantity: str, normal_period, refine: str = "parabola", mask=None, min_valid: float = 0.8) -> dict:
    raw = raw_index(field, quantity, refine, mask, min_valid).sel(year=slice(*normal_period))
    return {"mean": float(raw.mean()), "std": float(raw.std())}


def project(field: xr.DataArray, stats: dict, quantity: str, normal_period, refine: str = "parabola", mask=None, min_valid: float = 0.8):
    """standardized anomaly index; `raw_climatology` (own normal-period mean) as coordinate or attribute"""
    raw = raw_index(field, quantity, refine, mask, min_valid)
    clim = normal_period_mean(raw, normal_period)
    index = ((raw - clim) / stats["std"]).rename("index")
    if "member" in index.dims:
        index = index.assign_coords(raw_climatology=("member", np.full(index.sizes["member"], clim)))
    return index, clim


#---------------building the mask--------------#
def build_mask(stores, min_models: int = 3, min_fraction: float = 0.5) -> xr.Dataset:
    """
    points below the ground in at least `min_models` models; a point counts for a model where it is
    missing in more than `min_fraction` of the member-years that have data (one store per model)
    """
    count, models = None, []
    for path in stores:
        ds = xr.open_zarr(path)
        model = ds.attrs.get("model") or Path(path).name.rsplit("_", 1)[0]
        if model in models:
            continue
        models.append(model)
        da = ds[[v for v in ds.data_vars if not v.endswith("_global_mean")][0]].load()
        has = da.notnull().any(["lat", "lon"])
        frac = da.isnull().where(has).sum(["year", "member"]) / has.sum()
        count = (frac > min_fraction).astype(int) if count is None else count + (frac > min_fraction)
    count = count.drop_vars([c for c in count.coords if c not in count.dims])
    out = xr.Dataset({"mask": (count >= min_models).astype("i1"), "n_models": count.astype("i2")})
    out.attrs = {"description": f"below-ground points: missing in >{min_fraction:g} of the member-years in >={min_models} of {len(models)} models",
                 "models": ", ".join(models)}
    return out


if __name__ == "__main__":
    out_file, stores = Path(sys.argv[1]), sys.argv[2:]
    if out_file.exists():
        raise SystemExit(f"{out_file} exists - give a new name (masks are referenced by their checksum)")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    build_mask(stores).to_netcdf(out_file)
    print(f"wrote {out_file}")
