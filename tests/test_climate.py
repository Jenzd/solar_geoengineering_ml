import numpy as np
import pandas as pd
import pytest
import xarray as xr
from srm_ml.climate import aligned_tas, weighted_spatial_mean, regional_response
from srm_ml.ml import make_features, evaluate_ridge_by_year


def example():
    t=pd.date_range("2020-01-01",periods=72,freq="MS")
    coords={"time":t,"lat":[-60.,0.,60.],"lon":[0.,90.,180.]}
    ctl=xr.DataArray(np.ones((72,3,3))*288, dims=("time","lat","lon"),coords=coords, attrs={"units":"K"})
    sai=ctl - 1.5
    return sai,ctl


def test_weighted_mean_constant():
    sai,ctl=example()
    out=weighted_spatial_mean(sai-ctl)
    assert np.allclose(out.values,-1.5)


def test_regional_response_is_sai_minus_control():
    sai,ctl=example()
    out=regional_response(sai,ctl)
    assert np.allclose(out.delta_tas.sel(region="Global").values,-1.5)
    assert "Tropics (23.5S–23.5N)" in out.region.values


def test_wrong_grid_raises():
    sai,ctl=example()
    with pytest.raises(ValueError,match="lat grids differ"):
        aligned_tas(sai.assign_coords(lat=[-50.,0.,50.]),ctl)


def test_chronological_evaluation():
    sai,ctl=example()
    d=make_features(regional_response(sai,ctl),ctl)
    scores,preds=evaluate_ridge_by_year(d)
    assert scores["train_years"][1] < scores["test_years"][0]
    assert len(preds)>0
    assert scores["ridge_mae_C"] < 0.01


def test_unit_mismatch_rejected():
    sai, ctl = example()
    ctl.attrs["units"] = "degC"
    with pytest.raises(ValueError, match="units differ"):
        aligned_tas(sai, ctl)
