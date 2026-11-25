import numpy as np
import xarray as xr

from climate_scenarios_2050 import partition as pt

YEARS = np.arange(1990, 2061)
INIT, REF = 2020, 10


def test_window_change_simple():
    x = (YEARS - INIT).astype(float)              # 0 at init
    out = pt.window_change(x[None], YEARS, INIT, 5, np.array([1, 6, 40]), REF)
    ref = np.mean(np.arange(-9, 1))                # -4.5
    assert np.allclose(out[0, :2], [np.mean(np.arange(1, 6)) - ref, np.mean(np.arange(6, 11)) - ref])
    assert np.isnan(out[0, 2])                     # window ends after 2060


def synthetic(a=(1.0, 2.0, 3.0), b=(0.5, 1.0, 1.5), members=(2, 3, 4), eps=0.3):
    """forced: a_s * b_m * (t - init) after init (0 before); internal: +-eps in every year after init"""
    splits = {}
    for s, a_s in enumerate(a):
        model_ids, internal = [], []
        for m, n in enumerate(members):
            for r in range(n):
                model_ids.append(f"m{m}")
                internal.append(np.where(YEARS > INIT, eps * (1 if r % 2 == 0 else -1), 0.0))
        forced = np.array([a_s * b_m * np.clip(YEARS - INIT, 0, None) + 10 for b_m in b])   # +10: positive reference
        splits[f"s{s}"] = (
            xr.DataArray(np.array(internal), dims=("member", "year"), coords={"year": YEARS, "member": [f"x{s}_{k}" for k in range(len(model_ids))], "model_id": ("member", model_ids)}),
            xr.DataArray(forced, dims=("model_id", "year"), coords={"model_id": [f"m{m}" for m in range(len(b))], "year": YEARS}),
        )
    return splits, np.array(a), np.array(b), eps


def test_partition_exact():
    splits, a, b, eps = synthetic()
    ds = pt.partition_dataset(splits, {"init": INIT, "window_lengths": [5], "max_lead": 20}, REF, "model", False, {"name": "x"})
    lead = 3
    sub = ds.sel(window_length=5, lead=lead)
    c = np.mean(np.arange(lead, lead + 5))         # mean (t - init) over the window; reference: 0
    assert np.isclose(float(sub["variance"].sel(source="internal")), eps**2)
    assert np.isclose(float(sub["variance"].sel(source="model")), np.mean([np.var(a_s * b * c, ddof=1) for a_s in a]))
    assert np.isclose(float(sub["variance"].sel(source="scenario")), np.var(a * b.mean() * c, ddof=1))
    assert np.allclose(ds["fraction"].sum("source"), 1)
    assert np.allclose(ds["scenario_mean"].sel(window_length=5, lead=lead), a * b.mean() * c)
    assert float(ds["centre_year"].sel(window_length=5, lead=lead)) == INIT + lead + 2


def test_partition_relative_and_member_weighting():
    splits, a, b, eps = synthetic(members=(1, 1, 4))
    rel = pt.partition_dataset(splits, {"init": INIT, "window_lengths": [10], "max_lead": 5}, REF, "model", True, {"name": "x"})
    ab = pt.partition_dataset(splits, {"init": INIT, "window_lengths": [10], "max_lead": 5}, REF, "model", False, {"name": "x"})
    # the forced reference mean is 10 everywhere: relative = 10 * absolute (in %)
    assert np.allclose(rel["variance"], 100 * ab["variance"])
    assert rel.attrs["units"] == "%"
    rel2 = pt.partition_dataset(splits, {"init": INIT, "window_lengths": [10], "max_lead": 5}, REF, "model", True, {"name": "x", "units": "m/s"})
    assert rel2.attrs["units"] == "%"
    assert ab.attrs["units"] == ""
    # single-member models: one member at +eps each; +-eps alternate within the 4-member model -> eps^2 either way
    member = pt.partition_dataset(splits, {"init": INIT, "window_lengths": [10], "max_lead": 5}, REF, "member", False, {"name": "x"})
    assert np.allclose(member["variance"].sel(source="internal"), eps**2)


def test_box_mean_weights():
    lat, lon = np.array([0.0, 60.0]), np.array([0.0, 10.0])
    vals = np.zeros((1, 1, 2, 2))
    vals[..., 1, :] = 1.0                           # 1 at 60N
    da = xr.DataArray(vals, dims=("member", "year", "lat", "lon"), coords={"lat": lat, "lon": lon})
    out = pt.box_mean(da, {"lon": [-5, 15], "lat": [-5, 65]})
    assert np.isclose(float(out), 0.5 / 1.5)        # cos 60 = 0.5
