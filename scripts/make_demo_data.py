"""Generate SYNTHETIC climate arrays solely for testing the analysis pipeline.

These values are invented and MUST NOT be treated as GeoMIP data or research findings.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/demo")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    time = pd.date_range("2020-01-01", periods=180, freq="MS")
    lat = np.arange(-75.0, 76.0, 15.0)
    lon = np.arange(0.0, 360.0, 30.0)
    t = np.arange(len(time))[:, None, None]
    la = np.deg2rad(lat)[None, :, None]
    lo = np.deg2rad(lon)[None, None, :]
    seasonal = 5 * np.cos(2 * np.pi * t / 12) * np.sin(la)
    control = 285 - 40 * np.sin(la)**2 + seasonal + 0.027 * t + np.zeros_like(lo) + rng.normal(0, 0.18, (len(time),len(lat),len(lon)))
    # Fictional control-vs-intervention pattern for TESTING; not physically fitted.
    cooling = (-0.015 * t) * (0.8 + 0.2 * np.cos(la)) * (1 + 0.1 * np.cos(lo))
    sai = control + cooling + rng.normal(0, 0.12, control.shape)
    for field, scenario in ((control, "ssp585"), (sai, "g6sulfur")):
        da = xr.DataArray(field.astype("float32"), dims=("time","lat","lon"), coords={"time":time,"lat":lat,"lon":lon}, name="tas", attrs={"units":"K", "description":"SYNTHETIC DEMONSTRATION DATA, NOT GEOMIP"})
        ds = da.to_dataset()
        ds.attrs.update({"data_status":"SYNTHETIC DEMO ONLY — NOT SCIENTIFIC OBSERVATIONS", "scenario_label":scenario})
        path=output / f"{scenario}_tas_SYNTHETIC.nc"
        ds.to_netcdf(path, engine="scipy")
        print(f"Wrote {path}")

if __name__ == "__main__":
    main()
