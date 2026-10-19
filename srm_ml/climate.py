"""Climate response metrics for matched GeoMIP experiments.

These functions describe simulation output; they do not validate real-world SRM outcomes.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

REGIONS = {
    "Arctic (60–90N)": (60.0, 90.0),
    "Northern midlatitudes (23.5–60N)": (23.5, 60.0),
    "Tropics (23.5S–23.5N)": (-23.5, 23.5),
    "Southern midlatitudes (60–23.5S)": (-60.0, -23.5),
    "Antarctic (90–60S)": (-90.0, -60.0),
}


def validate_tas(da: xr.DataArray) -> None:
    """Require a gridded near-surface temperature variable with lat/lon/time."""
    missing = set(("time", "lat", "lon")) - set(da.dims)
    if missing:
        raise ValueError(f"tas is missing required dimensions: {sorted(missing)}")
    if da.sizes["time"] < 12:
        raise ValueError("Need at least 12 common monthly time steps for exploratory analysis")
    if da.sizes["lat"] < 2 or da.sizes["lon"] < 2:
        raise ValueError("Expected a nontrivial spatial grid")
    if "units" in da.attrs and da.attrs["units"] not in ("K", "kelvin", "Kelvin", "degC", "C", "celsius", "degree_Celsius"):
        raise ValueError(f"Unexpected tas units: {da.attrs['units']!r}")


def aligned_tas(sai: xr.DataArray, control: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Align paired monthly SAI/reference tas fields on identical grid and time."""
    validate_tas(sai)
    validate_tas(control)
    sai_units = sai.attrs.get("units")
    control_units = control.attrs.get("units")
    if sai_units and control_units and sai_units != control_units:
        raise ValueError(f"SAI/reference tas units differ: {sai_units} vs {control_units}. Convert before analysis.")
    for dim in ("lat", "lon"):
        if sai.sizes[dim] != control.sizes[dim] or not np.allclose(sai[dim].values, control[dim].values):
            raise ValueError(f"SAI/reference {dim} grids differ. Regrid explicitly before analysis.")
    sai2, control2 = xr.align(sai, control, join="inner")
    if sai2.sizes["time"] < 12:
        raise ValueError("SAI and reference scenarios have fewer than 12 common monthly time points")
    return sai2, control2


def weighted_spatial_mean(data: xr.DataArray, lat_range: tuple[float, float] | None = None) -> xr.DataArray:
    """Cos(latitude)-weighted mean; approximation for regular lat-lon grids.

    Do not use blindly on curvilinear grids; use cell-area variables like areacella.
    """
    if lat_range is not None:
        lo, hi = lat_range
        data = data.where((data.lat >= lo) & (data.lat <= hi), drop=True)
        if data.sizes.get("lat", 0) == 0:
            raise ValueError(f"No grid cells in latitude region {lat_range}")
    weights = np.cos(np.deg2rad(data.lat)).clip(min=0)
    return data.weighted(weights).mean(dim=("lat", "lon"))


def regional_response(sai: xr.DataArray, control: xr.DataArray) -> xr.Dataset:
    """Return regional SAI-reference temperature differences as a Dataset.

    Positive means warmer in SAI than the matched non-SAI reference simulation.
    For G6sulfur, the appropriate reference is normally SSP5-8.5 for the same model.
    """
    sai, control = aligned_tas(sai, control)
    diff = sai - control
    diff.attrs["units"] = "degC (difference)"
    names = ["Global"] + list(REGIONS)
    arrays = [weighted_spatial_mean(diff)]
    arrays.extend(weighted_spatial_mean(diff, region) for region in REGIONS.values())
    combined = xr.concat(arrays, dim=xr.IndexVariable("region", names))
    return xr.Dataset({"delta_tas": combined})
