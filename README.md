# Solar Geoengineering × Machine Learning

**Status: Work in progress — methodological prototype, not a scientific result.**

Exploratory Python research on how machine learning and statistical methods can characterize **simulated regional climate responses to stratospheric aerosol injection (SAI)**. The first focus is GeoMIP **G6sulfur vs the matched SSP5-8.5 non-SAI reference**, evaluating temperature differences and a simple ML baseline.

> **Evidence and reproducibility:** This initial repository documents the start of a genuine research effort; it does **not** claim completed GeoMIP experiments or validated findings. The included toy dataset is **synthetic**, generated solely for code testing and explicitly labeled as such in all output.

## Research questions

1. How do simulated near-surface temperature responses differ between G6sulfur and matched SSP5-8.5 runs, globally and by latitude band?
2. Can a simple regularized ML baseline predict these regional simulated responses on future, held-out years better than a training-period regional mean?
3. Later: how do performance and uncertainty change when climate models, ensemble members, or SRM scenarios are held out?

## Reproduce the synthetic code demonstration today

Requires Python 3.10+. Create a virtual environment and run:

```bash
python -m pip install -e '.[dev]'
python scripts/make_demo_data.py
python scripts/run_analysis.py \
  --sai data/demo/g6sulfur_tas_SYNTHETIC.nc \
  --control data/demo/ssp585_tas_SYNTHETIC.nc \
  --output outputs/demo --demo
pytest -q
```

The demonstration creates two figures, regional mean CSV, a temporal-holdout prediction CSV, and `run_metadata.json`. All are marked **synthetic**; they are not to be presented as GeoMIP findings.

## Analyze actual GeoMIP output (when available)

Get monthly `tas` for **G6sulfur and the matched SSP5-8.5 experiment** from an ESGF GeoMIP data portal. Select the same model, compatible realization, calendar, native grid, and time interval. Confirm inputs are K or °C and lat/lon coordinates. Then:

```bash
python scripts/run_analysis.py \
  --sai path/to/G6sulfur_tas_Amon.nc \
  --control path/to/ssp585_tas_Amon.nc \
  --output outputs/geomip-first-run
```

Outputs will remain **exploratory simulated results**, not evidence of real-world safety or effectiveness. The present code expects a single file per scenario with coordinate dimensions `time`, `lat`, `lon` and a `tas` variable. Concatenate multi-file CMIP6 outputs or regrid differing grids separately before use; it intentionally raises errors rather than silently mixing grids.

## Project structure

- `srm_ml/climate.py`: temperature pairing, regional weighted means
- `srm_ml/ml.py`: temporal-split Ridge baseline and training-period mean comparison
- `scripts/make_demo_data.py`: explicitly invented arrays for software tests
- `scripts/run_analysis.py`: scenario analysis, model evaluation, plot outputs
- `tests/`: unit tests for climate computations, chronology and grid consistency
- `docs/RESEARCH_PLAN.md`: scientific rationale, limitations, milestones, sources
- `docs/RESEARCH_LOG.md`: dated record of actual work; update honestly as progress occurs

## Scientific cautions

- This is a *baseline research prototype*, not a geoengineering deployment optimizer.
- Temporal prediction of one simulated response does not imply generalization to unseen SAI strategies.
- GeoMIP experiments have important model-specific assumptions and data-use requirements.
- Cos(latitude) weighting assumes a regular latitude–longitude grid; advanced analysis should use actual cell areas.
- Documentation and code were initially scaffolded with AI assistance. Original validation and interpretation remain future work.

## Acknowledgments and sources

- GeoMIP overview and data-use guidance: https://climate.envsci.rutgers.edu/GeoMIP/data.html
- GeoMIP G6 experiment specifications: https://climate.envsci.rutgers.edu/GeoMIP/geomip6.html
- Visioni et al. 2021, uncertainty in GeoMIP G6 experiments: https://acp.copernicus.org/articles/21/10039/2021/
- Community shared code: https://github.com/GeoMIP/Shareable-code

## License

MIT for original project code. External climate-model data retain their own licenses and policies and are not redistributed in this repository.
