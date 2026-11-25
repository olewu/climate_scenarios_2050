import numpy as np
import pandas as pd
import xarray as xr

from climate_scenarios_2050.core.timeconv import seasonal_mean
from climate_scenarios_2050.preprocess.indices import amv


def _field(values, lat=(0.0, 60.0)):
    return xr.DataArray(np.asarray(values, dtype=float), dims=("year", "lat", "lon"),
                        coords={"year": np.arange(len(values)), "lat": list(lat), "lon": [0.0, 1.0]})


def test_area_mean_weights_and_coverage():
    clim = xr.zeros_like(_field([[[0, 0], [0, 0]]]).isel(year=0))
    field = _field([[[1, 1], [3, 3]],                    # full coverage: cos-lat weights 1 and 0.5
                    [[1, np.nan], [np.nan, np.nan]],     # 1/3 of the area: below min_coverage
                    [[1, 1], [np.nan, np.nan]]])         # 2/3 of the area: mean of the valid points
    mean = amv.area_mean(field - clim, clim, min_coverage=0.5)
    np.testing.assert_allclose(mean.values, [(1 * 1 + 3 * 0.5) / 1.5, np.nan, 1.0])


def test_area_mean_ignores_points_without_climatology():
    clim = _field([[[0, np.nan], [0, 0]]]).isel(year=0)   # e.g. a land point
    field = _field([[[2, np.nan], [2, 2]]])
    assert float(amv.area_mean(field - clim, clim, min_coverage=1.0)) == 2.0


def test_seasonal_mean_min_months():
    time = pd.date_range("2000-01-01", periods=24, freq="MS")
    values = np.ones(24)
    values[:7] = np.nan            # 2000: 5 valid months, 2001: complete
    values[12] = 3.0
    da = xr.DataArray(values, dims="time", coords={"time": time})
    strict = seasonal_mean(da, list(range(1, 13)))
    assert np.isnan(strict.sel(year=2000)) and strict.sel(year=2001) > 1
    partial = seasonal_mean(da, list(range(1, 13)), min_months=5)
    assert float(partial.sel(year=2000)) == 1.0
    np.testing.assert_allclose(float(partial.sel(year=2001)), float(strict.sel(year=2001)))
    assert np.isnan(seasonal_mean(da, list(range(1, 13)), min_months=6).sel(year=2000))
