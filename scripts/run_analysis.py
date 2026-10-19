"""Run pilot SRM/reference comparison and temporal-split ML baseline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from srm_ml.climate import aligned_tas, regional_response
from srm_ml.ml import make_features, evaluate_ridge_by_year


def main():
    parser=argparse.ArgumentParser(description="Matched-scenario exploratory analysis. NOT SRM validation.")
    parser.add_argument("--sai", required=True, help="G6sulfur tas monthly NetCDF (or synthetic demo)")
    parser.add_argument("--control", required=True, help="matched SSP5-8.5 tas monthly NetCDF")
    parser.add_argument("--output", default="outputs/pilot")
    parser.add_argument("--demo", action="store_true", help="MANDATORY flag for synthetic demo data")
    args=parser.parse_args()
    out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    with xr.open_dataset(args.sai) as ds: sai=ds["tas"].load(); sai_synth="SYNTHETIC" in str(ds.attrs)
    with xr.open_dataset(args.control) as ds: control=ds["tas"].load(); ctl_synth="SYNTHETIC" in str(ds.attrs)
    if (sai_synth or ctl_synth) != args.demo:
        raise SystemExit("Synthetic files require --demo; --demo is forbidden with real inputs")
    sai,control=aligned_tas(sai,control)
    response=regional_response(sai,control)
    response.to_netcdf(out / "regional_response.nc",engine="scipy")
    df=make_features(response,control)
    scores,predictions=evaluate_ridge_by_year(df)
    predictions.to_csv(out / "ml_predictions.csv",index=False)
    response.delta_tas.to_dataframe(name="delta_tas_C").reset_index().to_csv(out / "monthly_regional_response.csv",index=False)
    global_response=response.delta_tas.sel(region="Global")
    fig,ax=plt.subplots(figsize=(8,4))
    ax.plot(global_response.time.values,global_response.values)
    ax.axhline(0,color="black",lw=0.7)
    ax.set(xlabel="Simulation year",ylabel="SAI minus reference (°C)",title="SYNTHETIC TEST DATA — NOT RESEARCH RESULTS" if args.demo else "Simulated global-mean temperature response")
    fig.tight_layout();fig.savefig(out/"global_temperature_delta.png",dpi=160);plt.close(fig)
    period=(sai-control).mean("time")
    fig,ax=plt.subplots(figsize=(8,4))
    im=ax.pcolormesh(period.lon,period.lat,period,cmap="RdBu_r",shading="auto")
    fig.colorbar(im,ax=ax,label="SAI minus reference (°C)")
    ax.set(xlabel="Longitude",ylabel="Latitude",title="SYNTHETIC TEST DATA — NOT RESEARCH RESULTS" if args.demo else "Simulated mean response over overlapping period")
    fig.tight_layout();fig.savefig(out/"spatial_temperature_delta.png",dpi=160);plt.close(fig)
    report={"status":"SYNTHETIC DEMO — NO CLIMATE FINDINGS" if args.demo else "EXPLORATORY MODEL ANALYSIS — NOT REAL-WORLD VALIDATION", "sai_input":str(args.sai),"reference_input":str(args.control),"common_months":int(sai.sizes["time"]),"method":"paired time/grid, cos(latitude) approximate weights, chronological train/test split", "ml_scores":scores,"important_limitations":["ML baseline has no aerosol intervention inputs", "No out-of-simulation or out-of-model generalization is established", "Compare only matching models, realizations, time periods and reference pathways", "Cos(latitude) is appropriate only for regular latitude-longitude grids"]}
    (out/"run_metadata.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    main()
