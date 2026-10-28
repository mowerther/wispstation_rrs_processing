# WISPstation Rrs processing

`wispstation-rrs` computes the remote sensing reflectance (Rrs) of WISPstation Level 1 (L1) measurements and states, for every measurement and every satellite overpass, whether the in situ side of the matchup is usable. The processing follows the WISPstation processing of Werther et al. (in revision): a fit of the three component model (3C), the Quality Water Index Polynomial (QWIP), an illumination criterion on the downwelling irradiance at 560 nm and band Rrs of the MultiSpectral Instrument (MSI) of Sentinel-2 and the Ocean and Land Colour Instrument (OLCI) of Sentinel-3.

One command processes a folder of L1 files:

```bash
wisp-rrs process path/to/l1 --products overpasses.txt --out results
```

## Contents

1. [What the processing does](#what-the-processing-does)
2. [Installation](#installation)
3. [Example run](#example-run)
4. [Inputs](#inputs)
5. [Options](#options)
6. [Outputs](#outputs)
7. [Quality criteria](#quality-criteria)
8. [The two 3C implementations](#the-two-3c-implementations)
9. [Band Rrs](#band-rrs)
10. [Verification](#verification)
11. [Data sources, licences and references](#data-sources-licences-and-references)

## What the processing does

The processing runs eight steps for each station found in the L1 folder.

1. **Read.** Every L1 file of the folder is read. Days split over several files are merged, and repeated measurement times are dropped. The station follows from the file name.
2. **Geometry and meteorology.** The solar zenith and azimuth angles come from pvlib for the station coordinates. Relative humidity (RH), air pressure and water temperature come from a meteo CSV file, or from defaults (RH 60 %, pressure from the station altitude, water temperature 20 °C).
3. **3C fit.** The 3C model (Groetsch et al., 2017) separates the sun and sky glint at the water surface from the water signal in every measurement. Two implementations can be selected: `jax` (Werther, the implementation of the paper processing) and `o25` (3C-O25, Pitarch, 2026). The fits run in parallel worker processes with a progress bar.
4. **QWIP.** The QWIP score (Dierssen et al., 2022) of the 3C Rrs measures how far a spectrum departs from the spectra expected for its apparent visible wavelength (AVW).
5. **Illumination.** The measured downwelling irradiance at 560 nm, Ed(560), is compared with a clear sky model of the day. The mean absolute percentage error (MAPE) over the 15 min around the overpass decides whether the overpass is retained.
6. **Matchup window.** Overpass times come from Sentinel-2 or Sentinel-3 product names, or from a CSV file of times. Every measurement within 15 min of an overpass (the default window) is paired with it.
7. **Band Rrs.** The hyperspectral 3C Rrs (350–900 nm, 1 nm) is resampled to the spectral response functions (SRF) of Sentinel-2A, 2B and 2C (S2A, S2B, S2C) and of Sentinel-3A and 3B (S3A, S3B).
8. **Tables and plots.** One table lists every measurement with one column per criterion. A second table lists every pair of overpass and measurement with the same columns and the matchup columns. A third table summarises each overpass. The illumination plot of each overpass follows Figure 2 of the paper: the clear sky model of the day with its envelope of plus or minus 20 %, the measurements in the matchup window in green (inside the envelope) or red (outside) and the other measurements of the day in orange, over the local time.

## Installation

The package needs Python 3.10 or later. With conda:

```bash
git clone https://github.com/mowerther/wispstation_rrs_processing.git
cd wispstation_rrs_processing
conda env create -f environment.yml
conda activate wispstation_rrs
```

The environment installs the package in editable mode with `pip install -e .[dev]`. In an existing Python environment, the same command in the repository folder installs the package and its dependencies (numpy, scipy, pandas, netCDF4, pvlib, lmfit, jax, tqdm, matplotlib).

## Example run

The example reproduces three matchup days of the paper from three L1 files in `examples/l1`, one per station. Times are in coordinated universal time (UTC) unless marked local.

| File | Station | Format | Day |
| --- | --- | --- | --- |
| `L1B_Lake_Greifen_Sprectral_20220618_040005.nc` | Greifensee (GFN) | Datalakes | 18 June 2022, S2B overpass at 10:25:59 UTC |
| `L1_Lexplore_Sprectral_20250528_060005.nc` | Lake Geneva, LéXPLORE platform (LXP) | Datalakes | 28 May 2025, S2C overpass at 10:26:21 UTC |
| `WISPstation012_20250710_L1_SPEC.nc` | Lake Trasimeno, WISPstation012 (TRS012) | WISPcloud | 10 July 2025, S2A overpass at 10:00:51 UTC |

Three files complete the example. `overpasses.txt` holds the Sentinel-2 product names of the three overpasses. `meteo.csv` holds the RH, pressure and water temperature of the 3C fits of the paper processing. `edmax.csv` holds the clear sky maximum of Ed(560) of each day (EdMax, see [Illumination](#illumination)).

```bash
wisp-rrs process examples/l1 --out example_output --products examples/overpasses.txt --meteo examples/meteo.csv --edmax examples/edmax.csv
```

The run fits 128 measurements in about 20 s with two workers. Two of the three overpasses have a usable measurement:

| Overpass | MAPE in the window | MAPE of the day | Decision | Usable in the window |
| --- | --- | --- | --- | --- |
| Greifensee, S2B, 2022-06-18 | 2.3 % | 6.5 % | retain | 2 of 2 |
| Lake Geneva, S2C, 2025-05-28 | 80.8 % | 70.6 % | reject | 0 of 2 |
| Lake Trasimeno, S2A, 2025-07-10 | 0.5 % | 21.5 % | retain | 2 of 2 |

The Lake Geneva measurements in the window also fail the QWIP criterion, with scores of 0.39 and 0.57. The option `--threec o25` repeats the run with 3C-O25.

## Inputs

The processing needs a folder of L1 files and, for matchups, the overpass times. A meteo file and an EdMax file are optional.

### L1 files

Two formats are read:

| Format | Stations | File names | Variables |
| --- | --- | --- | --- |
| Datalakes | Greifensee, Lake Geneva | `L1B_Lake_Greifen_Spectral_*`, `L1_Lexplore_Spectral_*`, also spelled `Sprectral` | `downirr` (Ed), `downrad` (sky radiance Ls), `uprad` (upwelling radiance Lu), quality masks `*_qual` |
| WISPcloud | Lake Trasimeno, WISPstation012 (TRS012) and WISPstation001 (TRS001) | `WISPstationNNN_YYYYMMDD_L1_SPEC.nc` | `downirr`, `downrad`, `uprad`, quality flags `*_qc` |

The folder is searched recursively. The station table in `src/wispstation_rrs/stations.py` holds the coordinates, altitude, time zone, sensor geometry (view zenith angle 45°, relative azimuth 135°) and Sentinel-2 tiles of the four stations.

### Another station

Files of a station outside the station table are read with its code, coordinates and time zone:

```bash
wisp-rrs process path/to/l1 --station ABC --lat 47.37 --lon 8.66 --altitude 435 --timezone Europe/Zurich --tiles T32TMT --products overpasses.txt
```

In Python, `custom_station` of `wispstation_rrs.stations` builds the station, and `process(..., station=custom_station("ABC", 47.37, 8.66, 435, timezone="Europe/Zurich", tiles=("T32TMT",)))` reads every file of the folder as that station. The view zenith angle (45°) and relative azimuth (135°) are arguments of `custom_station`.

### Overpass times

- `--products`: a text file with one product name per line, a folder of products (`.SAFE`, `.SEN3`, `.zip`) or names on the command line. Sentinel-2 names give the sensing start time and the tile (`S2A_MSIL1C_20250710T100051_N0511_R122_T32TQN_...`). Sentinel-3 OLCI names give the start and stop times of the acquisition, and the overpass time is their midpoint (`S3A_OL_1_EFR____<start>_<stop>_...`). Products of one acquisition on several tiles count as one overpass.
- `--times`: a CSV file with the column `time_utc` and the optional columns `platform` (for example `S2A` or `S3B`), `label` and `station`.

A Sentinel-2 product is paired with the stations whose tiles include its tile. Sentinel-3 products and times carry no tile, and each of them is paired with every station of the folder. In a text file of product names, a station code after the name (`S3A_OL_1_EFR____... GFN`) restricts the product to that station, and the column `station` of a times file does the same. Without overpass times, the illumination criterion uses a window centred on each measurement, and the pairs table and the overpass summary are not written.

`wisp-rrs overpasses examples/overpasses.txt` lists the overpass times that the processing reads from product names.

### Meteo file

`--meteo` reads a CSV file with the columns `time_utc`, `station` (optional), `rh_percent`, `pressure_hpa` and `water_temperature_c`. Each measurement takes the nearest row of its station within 90 min. A measurement without a row takes the defaults of `--rh`, `--pressure` and `--water-temp`. The column `meteo_source` of the output states which source was used.

### EdMax file

`--edmax` reads a CSV file with the columns `station`, `date_local` and `ed_max`. Its values replace the EdMax computed from the L1 files for those dates. A row without a station applies to every station. The file `illumination_daily.csv` of an earlier run has these columns.

## Options

Every option has a default, and the defaults of the criteria and windows are the settings of the paper.

| Option | Default | Meaning |
| --- | --- | --- |
| `--out` | `wisp_rrs_output` | Output folder |
| `--threec` | `jax` | 3C implementation, `jax` or `o25` |
| `--station` | from the file names | Station code for all files |
| `--lat`, `--lon`, `--altitude`, `--timezone`, `--tiles` | none | A station outside the station table, with `--station` |
| `--products`, `--times` | none | Overpass times |
| `--window` | 15 | Matchup window in minutes |
| `--illum-window` | 15 | Window of the illumination MAPE in minutes |
| `--context` | 60 | Measurements within this many minutes of an overpass enter the pairs table |
| `--fit` | `all` | Fit all measurements, or only those in a matchup window (`window`) |
| `--meteo` | none | Meteo file |
| `--rh`, `--water-temp`, `--pressure` | 60 %, 20 °C, from the altitude | Defaults for measurements without a meteo row |
| `--mape-max` | 20 | Limit of the illumination MAPE in % |
| `--edmax` | none | EdMax file |
| `--qwip-max` | 0.2 | Limit of the absolute QWIP score |
| `--sza-max` | none | Limit of the solar zenith angle in degrees |
| `--sensors` | S2A S2B S2C S3A S3B | Sensors of the band Rrs |
| `--min-coverage` | 0.99 | Share of a band response that the spectrum must cover |
| `--workers` | 2 | Worker processes of the 3C fits |
| `--no-resume` | off | Fit again instead of taking the 3C results of an earlier run |
| `--no-plots` | off | Write no plots |
| `--quiet` | off | No progress bar |

The 3C results of each station and day are kept in `<out>/cache`. A second run with the same files, inputs and 3C settings takes them from there, and an interrupted run continues with the days that were not finished.

`wisp-rrs plot <out>` draws the plots of an output folder again. The same processing runs from a Python script. The worker processes import the script again, so the call sits under `if __name__ == "__main__":`

```python
from wispstation_rrs.batch import Settings, process

if __name__ == "__main__":
    tables = process("examples/l1", "example_output", Settings(threec="jax", workers=2),
                     products="examples/overpasses.txt", meteo="examples/meteo.csv", edmax="examples/edmax.csv")
    print(tables["summary"][["station", "platform", "illum_decision", "matchup_usable"]])
```

## Outputs

A run writes five tables, the run settings and the plots to the output folder.

| File | Rows | Content |
| --- | --- | --- |
| `insitu_measurements.csv` | one per measurement | Geometry, meteo inputs, criteria, 3C parameters, QWIP, Ed(560), band Rrs, `insitu_usable`, `reason` |
| `insitu_overpass_pairs.csv` | one per overpass and measurement within `--context` | Matchup columns and all columns of the measurements table |
| `overpass_summary.csv` | one per overpass with measurements | Products and tiles, counts in the window, illumination decision, closest usable measurement and its band Rrs for the platform of the overpass |
| `rrs_hyperspectral.csv` | one per measurement | 3C Rrs at 350–900 nm in sr⁻¹, one column per nanometre (`rrs_350`, …, `rrs_900`) |
| `illumination_daily.csv` | one per station and local date | EdMax and its source, MAPE of the day, sunrise and sunset |
| `run_settings.json` | one file | Settings, input files and counts of the run |
| `plots/` | one file per plot | Illumination per overpass, measured against modelled Ed(560), QWIP diagram and spectra in the window |

Floating point values are written with 12 significant digits.

### Columns of the measurements table

| Group | Columns |
| --- | --- |
| Identity | `station`, `file`, `time_utc`, `time_local`, `date_local` |
| Geometry and meteo | `sza`, `saa` (solar zenith and azimuth), `vza`, `raa` (view zenith and relative azimuth), `rh_percent`, `pressure_hpa`, `water_temperature_c`, `meteo_source` |
| L1 criteria | `ed_qual_ok`, `ls_qual_ok`, `lu_qual_ok`, `qual_ok`, `spectra_finite`, `spectra_positive`, `sza_ok`, `l1_ok` |
| 3C | `threec_model`, `threec_fitted`, `threec_ok`, `threec_converged`, `threec_cost`, `threec_nfev`, `threec_message`, parameters `p_*` |
| QWIP | `avw_nm`, `ndi`, `qwip_score`, `qwip_ok` |
| Illumination | `ed560`, `ed560_model`, `ed560_rel_error_percent`, `ed_max`, `illum_n_points`, `illum_mape`, `illum_reason`, `illumination_ok` |
| Result | `insitu_usable` (all criteria met), `reason` (the criteria not met, joined by `+`, or `OK`) |
| Band Rrs | `rrs_<SENSOR>_<BAND>`, for example `rrs_S2A_B4` or `rrs_S3B_Oa08` |

In the measurements table, the illumination window is centred on the measurement. In the pairs table, the window is centred on the overpass, as in the paper, and the columns `illum_*` and `illumination_ok` refer to the overpass.

### Columns of the pairs table

`overpass_id`, `product`, `platform`, `sensor`, `overpass_time_utc`, `delta_min` (measurement time minus overpass time), `abs_delta_min`, `in_window`, `is_closest` (nearest measurement in the window), `is_closest_usable` (nearest usable measurement in the window), `insitu_usable`, `reason`, `illum_decision`, `illum_reason_overpass`, `illum_mape_window`, `illum_mape_day`, `illum_n_points_window`, `illum_ed_max`, then all columns of the measurements table.

The measurements in the window that are not usable, with the criteria they do not meet:

```python
import pandas as pd

pairs = pd.read_csv("example_output/insitu_overpass_pairs.csv")
print(pairs.loc[pairs["in_window"] & ~pairs["insitu_usable"], ["station", "time_utc", "delta_min", "reason"]])
```

In the example, the two Lake Geneva measurements of 28 May 2025 have the reason `QWIP+ILLUMINATION`.

## Quality criteria

A measurement is usable when it meets every criterion of the table. Each criterion has its own column, and `reason` names the criteria not met.

| Column | Criterion | Reason code | Default |
| --- | --- | --- | --- |
| `qual_ok` | The L1 quality flags of Ed, Ls and Lu are zero | `L1_FLAG` | on |
| `spectra_finite` | Ed, Ls and Lu are finite at 350–900 nm | `L1_NOT_FINITE` | on |
| `spectra_positive` | Ed, Ls and Lu are positive at 350–900 nm | `L1_NOT_POSITIVE` | on |
| `sza_ok` | Solar zenith angle at most `--sza-max` | `SZA` | no limit |
| `threec_ok` | The 3C fit gives finite Rrs at every wavelength | `THREEC` | on |
| `qwip_ok` | Absolute QWIP score at most `--qwip-max` | `QWIP` | 0.2 |
| `illumination_ok` | Ed(560) MAPE in the window at most `--mape-max` | `ILLUMINATION` | 20 % |

Measurements with spectra that are not finite or not positive, and measurements with the sun below the horizon, are not fitted. The column `threec_converged` holds the convergence flag of the optimiser for information. The paper processing kept every fit, and the criterion `threec_ok` follows it.

### QWIP

AVW is the harmonic mean wavelength over 400–700 nm, weighted by Rrs. The normalised difference index (NDI) is (Rrs(665) − Rrs(492)) / (Rrs(665) + Rrs(492)), at the bands nearest 492 and 665 nm within 2 nm. The QWIP score is the polynomial of Dierssen et al. (2022) at the AVW minus the NDI, computed from the 3C Rrs.

### Illumination

The illumination criterion compares the measured Ed(560) with a clear sky model of the day. The settings are those of the paper:

- Ed at the band nearest 560 nm, within 5 nm.
- EdMax of a date is the 90th percentile of the Ed(560) values measured within 15 min of solar noon, pooled over the dates within 15 days before and after.
- The model is EdMax · sin(π t / L)^1.8, with t the time after sunrise minus 8 min and L the length of the day. Sunrise, noon and sunset follow the solar equations of the National Oceanic and Atmospheric Administration (NOAA).
- The overpass is retained when the MAPE between measured and modelled Ed(560) over the measurements within 15 min of the overpass (`--illum-window`) is at most 20 %, with at least one measurement.

EdMax needs the noon measurements of the days around a date. A run over a full station archive computes EdMax from its own files. The three example files hold one day each, and EdMax would then rest on that day alone. On 28 May 2025, Lake Geneva has no measurement within 15 min of solar noon, and no EdMax follows from the file. The example therefore passes `examples/edmax.csv`, with the EdMax of the three days computed from the full archives. With that file, the example run gives the MAPE values of the paper tables, 2.311 % at Greifensee and 0.499 % at Lake Trasimeno.

## The two 3C implementations

Two 3C implementations are included, and `jax` is the default.

### `jax`: 3C JAX implementation (Werther)

The JAX implementation of the 3C model computes the gradient of the cost with automatic differentiation and fits with the L-BFGS-B method of lmfit. The copy in `src/wispstation_rrs/threec/jax_werther` is the 3C fit of the paper processing. The JAX version is also available at https://gitlab.com/pgroetsch/rrs_model_3C/-/tree/werther_jax_version.

| Setting | Value |
| --- | --- |
| Fitted parameters, start value and bounds | beta 0.5 (0.05–7), alpha 1.2 (0.1–2.5), C_chl 5 (0.01–100), C_sm 1 (0.01–100), C_y 0.5 (0.01–5), rho_dd 0 (0–0.1), rho_ds 0.01 (0–0.1) |
| Fixed parameters | air mass type 1, rho_s 0.0256, n_w 1.34, S_y 0.018, C_mie 0, delta 0 |
| Inputs per measurement | solar zenith angle, view zenith angle 45°, RH, pressure (bounds 950–1050 hPa), water temperature (bounds 0–35 °C) |
| Weights | 5 at 350–500 nm, 0.1 at 675–750 and 760–770 nm, 1 elsewhere |
| Optimiser | lmfit `lbfgsb`, gtol 1e-16, ftol 1e-16, eps 1e-7, maxiter 15000, maxls 20, maxcor 20 |
| Rrs | Lu/Ed minus the modelled sun and sky glint |

lmfit clips an input outside its bounds to the bound, and the column `p_pressure` holds the pressure of the fit. The barometric default falls below 950 hPa above about 540 m. The cost and its gradient are compiled once per worker. The results are bitwise those of the code path of the paper processing, which compiles them for every fit, and the runs are five to ten times faster. The [Verification](#verification) table gives the agreement with the archived Rrs.

### `o25`: 3C-O25 (Pitarch, 2026)

3C-O25 version 1.0.4 (https://github.com/jaipipor/3C-Rrs-O25, commit 54782ca) is included in `src/wispstation_rrs/threec/o25_pitarch` without changes to the model code or its data. The configuration follows the time series example of 3C-O25: parameter set and weights of the example, air mass type 4, the `leastsq` method of lmfit, and the view zenith and relative azimuth of the station. 3C-O25 does not use the water temperature. The retrieved Rrs is Lu/Ed minus the modelled glint Rg.

On the example days, the Rrs of the two implementations differ by up to 9 × 10⁻³ sr⁻¹ at 400–700 nm, and by up to 5 × 10⁻³ sr⁻¹ for the measurements in the matchup windows. QWIP and band values depend on the implementation chosen, and `run_settings.json` records it.

## Band Rrs

The band value of a measurement is the integral of the SRF times the Rrs, divided by the integral of the SRF. Both integrals run over the union of the measured wavelengths and the SRF wavelengths with trapezoid weights. A band with less than 99 % of its response within 350–900 nm stays empty (MSI B9–B12, OLCI Oa19–Oa21). MSI B8 has 99.1–99.5 % of its response within 900 nm, and its value is normalised over that part.

| Sensor | Source |
| --- | --- |
| S2A, S2B, S2C | Sentinel-2 MSI spectral response functions of the European Space Agency (ESA), document COPE-GSEG-EOPG-TN-15-0007, version 4.0 |
| S3A, S3B | ESA Sentinel-3 OLCI mean spectral response functions of S3A (13 July 2016) and S3B (9 January 2018) |

No OLCI response file of S3C is included. A file `olci_s3c.csv` in the layout of `olci_s3a.csv`, placed in `src/wispstation_rrs/srf/data`, adds S3C to the processing. In the same folder, `SOURCES.md` describes the files and `manifest.json` lists their checksums.

## Verification

The suite in the folder `tests` compares each step with the paper processing and runs in about one minute:

```bash
pytest
```

| Part | Reference | Agreement |
| --- | --- | --- |
| 3C JAX | Archived Rrs of the paper processing, 12 measurements of the three example days, with the archived inputs | within 1.4 × 10⁻⁷ sr⁻¹ at Greifensee (archive with three significant digits), 2 × 10⁻¹¹ sr⁻¹ at Lake Geneva and Lake Trasimeno |
| Full run with pvlib geometry and the example meteo file | Same archived Rrs at Lake Trasimeno | within 10⁻⁷ sr⁻¹ |
| 3C-O25 | Upstream version 1.0.4, on its example spectrum and one Lake Geneva measurement | bitwise equal |
| Solar zenith angle | Angles of the paper processing | within 0.006° |
| Sunrise and sunset | Daily tables of the paper | equal to the second |
| Illumination | MAPE of the paper tables, Greifensee on 18 June 2022 and Lake Trasimeno on 10 July 2025 | equal to 10⁻¹² |
| QWIP | QWIP functions of the paper processing | within 10⁻¹² |
| Band Rrs | Band values of the paper processing, MSI B1–B7 and B8A | within 10⁻⁹ sr⁻¹ |

## Data sources, licences and references

The example data come from two archives, and the code combines three licences.

### Data

- Greifensee and Lake Geneva (LéXPLORE platform, Wüest et al., 2021): WISPstation L1 data of Datalakes, https://www.datalakes-eawag.ch.
- Lake Trasimeno: WISPstation L1 data of the WISPcloud of Water Insight, from the stations of CNR-IREA (Institute for Electromagnetic Sensing of the Environment, National Research Council of Italy). The measurements are available from CNR-IREA upon request. The example file serves the example run only.
- WISPstation radiometer: Peters et al. (2018).

### Licences

The code is distributed under the GNU General Public Licence, version 3 or later (`LICENSE`). The 3C-O25 copy keeps its licence, GPL-3.0-or-later (`src/wispstation_rrs/threec/o25_pitarch/NOTICE.txt`). The JAX implementation derives from the 3C implementation of Groetsch, distributed under LGPL-3.0-or-later (`src/wispstation_rrs/threec/jax_werther/NOTICE.txt`).

### References

- Dierssen, H. M., Vandermeulen, R. A., Barnes, B. B., Castagna, A., Knaeps, E., and Vanhellemont, Q. (2022). QWIP: A quantitative metric for quality control of aquatic reflectance spectral shape using the apparent visible wavelength. Frontiers in Remote Sensing, 3, 869611. https://doi.org/10.3389/frsen.2022.869611
- Groetsch, P. M. M., Gege, P., Simis, S. G. H., Eleveld, M. A., and Peters, S. W. M. (2017). Validation of a spectral correction procedure for sun and sky reflections in above-water reflectance measurements. Optics Express, 25(16), A742–A761. https://doi.org/10.1364/OE.25.00A742
- Peters, S., Laanen, M., Groetsch, P., Ghezehegn, S., Poser, K., Hommersom, A., De Reus, E., and Spaias, L. (2018). WISPstation: A new autonomous above-water radiometer system. Proceedings of Ocean Optics XXIV, Dubrovnik. https://doi.org/10.5281/zenodo.2533079
- Pitarch, J. (2026). A general model for sun and sky glint removal in above-water optical radiometry: mathematical description and Python code. Earth Science Informatics, 19, 78. https://doi.org/10.1007/s12145-026-02114-w
- Pitarch, J., Talone, M., Zibordi, G., and Groetsch, P. (2020). Determination of the remote-sensing reflectance from above-water measurements with the 3C model: a further assessment. Optics Express, 28(11), 15885–15906. https://doi.org/10.1364/OE.388683
- Werther, M., Bresciani, M., Giardino, C., Irani Rahaghi, A., Maire, L., Runnalls, J., and Odermatt, D. (in revision). Assessment of adjacency effect correction methods for Sentinel-2 MSI observations over four European lakes.
- Wüest, A., Bouffard, D., Guillard, J., Ibelings, B. W., Lavanchy, S., Perga, M.-E., and Pasche, N. (2021). LéXPLORE: A floating laboratory on Lake Geneva offering unique lake research opportunities. WIREs Water, 8(5), e1544. https://doi.org/10.1002/wat2.1544

Author: Mortimer Werther
