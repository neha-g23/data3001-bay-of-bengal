# Seasonal reversal of Bay of Bengal surface currents

DATA3001 group project (UNSW, 2026). We map how surface currents in the Bay of Bengal change between the Southwest (June–September) and Northeast (December–February) monsoons, using hourly NOAA Global Drifter Program data from July 2000 to June 2022 (5–23°N, 78–99°E). Every figure, table and number in the final report can be regenerated from this repository.

## Repository structure

| Path | What it does |
|---|---|
| `project3001.py` | Main pipeline: downloads and subsets the data with clouddrift, cleans it, builds the drifter-weighted 1° grid, and writes the EDA figures and the unsmoothed grid estimate to `outputs/` |
| `reversal_figures.py` | Kernel smoothing (75 km seasons, 100 km months), coverage rule, direction change, and the seasonal and monthly maps |
| `method_comparison/` | Grid vs kernel smoothing vs kriging on held-out drifters and held-out years (report Table W); see its README |
| `report_code/` | One numbered script per report figure and table, with its own README listing the command for each output. Also holds `reversal_uncertainty.py` (whole-drifter bootstrap and sensitivity tests) and `baseline_vs_kernel.py` (1° grid vs kernel smoothing on held-out drifters) |

## Setup

Python 3.13 with the packages in `requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Getting the data

The data are not stored here (about 70 MB). Running `python project3001.py` downloads the hourly drifter data through clouddrift, subsets it to the Bay of Bengal and saves `bob_subset.nc`. Copy or point to that file for the other scripts (`--data bob_subset.nc`, or `BOB_DATA=/path/to/bob_subset.nc`).

## Reproducing the report

| Report item | Command |
|---|---|
| EDA (Figure E1, Table E1), unsmoothed 1° grid (67%), Figures M1–M2 | `python project3001.py` and `python report_code/09_eda_figures.py` |
| Seasonal maps (Figure M4), direction change (Figure R) | `python reversal_figures.py --data bob_subset.nc --out figures`, `python report_code/02_direction_change_map.py` |
| Bandwidth choice (Table M1) | `python report_code/13_bandwidth_cv.py` |
| Kernel vs kriging diagram (Figure M3) | `python report_code/14_kriging_diagram.py` |
| Headline 82% (95% CI 72–88%), consistency map (Figure U) | `python report_code/01_headline_bootstrap.py` |
| Regions (Table X, Figure X) | `python report_code/03_region_table.py`, `04_region_map.py` |
| Distance from the coast (Table Y, Figure W), speed effect (Figure S) | `07_q2_coast_distance.py`, `08_q2_speed_turn.py`, `11_q2_speed_turn_figure.py` |
| Monthly switch (Figure Z, Table V) | `05_monthly_switch.py`, `15_monthly_turn_table.py` |
| Sensitivity tests (Table Z) | `python report_code/12_sensitivity_intervals.py` |
| Method comparison (Table W) | `PYTHONPATH=report_code:method_comparison/scripts python method_comparison/scripts/kyear.py` (see its README) |

`bash report_code/run_all.sh` runs every report script in order (30–60 minutes, mostly the bootstraps). All random draws use seed 3001, so reruns give the same numbers; set `NB=20` for a quick check with fewer redraws.

## Reuse

The study region, season months, smoothing distance, coverage rule (3 drifters within 100 km) and speed threshold (0.1 m/s) are set at the top of `reversal_figures.py`. Changing them maps another region or season without editing the functions.

## Limitations

Distances from the coast use a simplified hand-drawn coastline of India's east coast and Sri Lanka; white map areas are places no drifter visited, not a true land mask; and undrogued drifters (used only as a sensitivity check) are also pushed by the wind.
