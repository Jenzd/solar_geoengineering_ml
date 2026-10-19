# Research plan (exploratory; no findings yet)

## Initial question
Can simple machine-learning models reproduce the simulated regional temperature response to stratospheric aerosol injection, and how does performance change under chronological holdout?

## Data
GeoMIP G6sulfur monthly near-surface temperature (`tas`) paired with the **same model's** SSP5-8.5 `tas` output; match realizations, overlapping periods, units, calendars, grids.

The GeoMIP G6sulfur experiment is designed to offset SSP5-8.5 forcing toward the SSP2-4.5 level. **Do not compare G6sulfur against SSP2-4.5 as a no-SAI counterfactual for the injection effect.**

## First baseline
1. Matched simulated temperature differences (`G6sulfur - SSP5-8.5`), with latitude-weighted regional and global means.
2. A chronological holdout, fitting a regularized regression against time, season, climate reference global temperature and region.
3. Compare against training-period regional mean baseline.

## Limitations that matter
- The initial model has no aerosol-emissions or intervention input, so it is **not a general climate emulator**.
- Strong climate autocorrelation motivates block time splits and eventually held-out ensemble members/models.
- Temperature responses from different models are not interchangeable training examples without harmonization.
- For curvilinear grids, replace latitude weights with proper land/ocean area-weight variables (`areacella` where available).
- Climate simulations are uncertain scenarios, not observations of deployed solar geoengineering.

## Next steps
- Obtain and document a matched pair of real G6sulfur and SSP5-8.5 outputs via ESGF.
- Reproduce one published area-weighted result.
- Add anomaly/ensemble uncertainty, compare multiple Earth system models, and leave-one-model-out evaluation.
- Develop out-of-distribution diagnostics or uncertainty calibration once enough independent simulations are available.
- Seek scientific feedback from solar-geoengineering collaborators and document contributions truthfully.

## Source and data policies
- GeoMIP data overview: https://climate.envsci.rutgers.edu/GeoMIP/data.html
- GeoMIP Phase 6 experiments: https://climate.envsci.rutgers.edu/GeoMIP/geomip6.html
- Shared analysis code: https://github.com/GeoMIP/Shareable-code
- Visioni et al. (2021): https://acp.copernicus.org/articles/21/10039/2021/
- Review GeoMIP model-specific data use/coauthorship policies before publication.
