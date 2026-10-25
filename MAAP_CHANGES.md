# Changes from upstream

Upstream: https://github.com/opera-adt/synth, branch `main` at commit 4cca7f1348c0d68ae8c29f5f91ceb9eaad4dd619 (2025-10-28), by Scott Staniewicz (JPL, OPERA project), Apache License 2.0 (`LICENSE`). The newest release is `v0.1.2` (3bbf728); `main` adds only CI updates on top of it.

This copy starts from upstream's file tree at 4cca7f1 without its git history. It is unchanged except for the items below. The simulation code in `src/synth/` and the LICENSE are untouched. Left out: `.github/` (upstream's CI, CD and dependabot settings).

## 2026-10-25: gdal is declared

The global coherence mode (the default whenever a configuration has no `custom_covariance`, for example `demos/global_coherence_sim.json`) builds its seasonal coherence layers in `global_coherence.calculate_seasonal_coeffs_files()` with GDAL's `gdal_calc` command-line tool. That tool comes with the conda-forge `gdal` package, which was not listed: `rasterio` brings only `libgdal-core`. So such a run downloaded the coherence rasters and then stopped at the first `gdal_calc` call with `/bin/sh: 1: gdal_calc: not found` (exit 127). The tests use a custom coherence model, so CI never reached this code.

`gdal` is now in `[tool.pixi.dependencies]` in `pyproject.toml` and in `environment.yml`, and `pixi.lock` was updated with pixi 0.48.2 (the version CI uses). The only package added to the lock is `gdal` 3.10.3, built against the `libgdal-core` 3.10.3 that was already locked; nothing else moved. It is not added to the PyPI dependencies, because the PyPI `gdal` package has to be compiled against a system GDAL.

Results do not change: custom coherence runs never used GDAL's command-line tools, and global coherence runs could not finish before.

## 2026-10-25: README install section

The README had no install steps. It now gives the pixi and conda routes and says that the global coherence mode needs `gdal_calc`, which a plain `pip install synth-insar` does not provide.

## Not changed

- `core.py` passes the mean backscatter from `global_coherence.convert_to_float()` to `make_noisy_samples_jax(..., amplitudes=amps)`, which multiplies the unit-power samples by it. The value is backscatter power (DN^2 / 199526231, as the dataset documents), so the simulated |SLC|^2 is power squared. The README's math section defines the multiplier as sigma = sqrt(E|x|^2), which would be the square root of power, while its workflow list says "multiply by the average backscatter". It only scales each pixel's brightness (the dB contrast between pixels doubles); phase, coherence and the CRLB files do not depend on it. Left as it is.
- `pyproject.toml` points `license` at `LICENSE.txt`, but the file is `LICENSE`. A wheel still builds.

## Check

On 2026-10-25, from a fresh local clone of this repository on Linux x86-64 (2 cores, 7 GB):

```sh
pixi install --locked            # pixi 0.48.2; default env: Python 3.13.5, jax 0.7.0 (CPU), gdal and libgdal-core 3.10.3
pixi install --locked -e test
pixi run --frozen -e test test -vv
pixi run --frozen synth-run -f tests/sample.json     # in an empty folder, no network
pixi run --frozen synth-run -f gc_small.json         # global coherence, see below
```

- Tests: 4 passed.
- `tests/sample.json` (custom coherence): 21 s, 0.45 GB peak memory, 266 MB of output. The coherence estimated from the simulated stack (11 by 11 looks) matches the requested model exp(-t / 60 days) at every date (0.820 vs 0.819 at 12 days, 0.373 vs 0.368 at 60 days), as before the change.
- `gc_small.json` is `demos/global_coherence_sim.json` with the bounding box of `tests/sample.json` (-117.9, 34.9, -117.75, 35.1), 6 dates and `output_dir` ".". It now runs to the end: 63 s, 1.45 GB peak memory, 431 MB of output, reading the seasonal coherence tiles anonymously over HTTPS from the AWS Open Data bucket `sentinel-1-global-coherence-earthbigdata`. The estimated coherence follows the model map (median 0.405 vs 0.393 at 12 days; correlation 0.88 at 12 days to 0.78 at 60 days), and the output amplitude follows the mean backscatter layer (r = 0.988).
- The conda route (`conda env create -f environment.yml`, which picked Python 3.15.1 and GDAL 3.13.3, then `pip install -e .`) also has `gdal_calc` and gives the same global coherence statistics: 61 s, 0.75 GB.

The turbulence and phase ramp layers are drawn without a fixed seed, so pixel values differ between runs; the statistics above do not.
