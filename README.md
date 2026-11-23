# Seasonal reversal of Bay of Bengal surface currents

DATA3001 group project (UNSW, 2026). We map how surface currents in the Bay of Bengal change between the Southwest (June–September) and Northeast (December–February) monsoons, using hourly NOAA Global Drifter Program data from July 2000 to June 2022 (5–23°N, 78–99°E). Every figure, table and number in the final report can be regenerated from this repository.

## Repository structure

| Path | What it does |
|---|---|
| `project3001.py` | Main pipeline: downloads and subsets the data with clouddrift, cleans it, builds the drifter-weighted 1° grid, and writes the 1° grid map (Figure M1) and the unsmoothed grid estimate to `outputs/`; see below for what else the report uses from it |
| `reversal_figures.py` | Kernel smoothing (75 km seasons, 100 km months), coverage rule, direction change, and the seasonal and monthly maps |
| `method_comparison/` | Grid vs kernel smoothing vs kriging on held-out drifters and held-out years (report Table W); see its README |
| `report_code/` | One script per report figure and table, named by report section (e.g. `3_4_...`, `4_1_q1_...`), with a README mapping every report item to its script. Also holds `reversal_uncertainty.py` (whole-drifter bootstrap and sensitivity tests) |

## Setup

Python 3.13 with the packages in `requirements.txt`:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Getting the data

The data are not stored here (about 70 MB). Running `python project3001.py` downloads the hourly drifter data through clouddrift, subsets it to the Bay of Bengal and saves `bob_subset.nc`. Copy or point to that file for the other scripts (`--data bob_subset.nc`, or `BOB_DATA=/path/to/bob_subset.nc`).

## Reproducing the report

```bash
python project3001.py                     # download data -> bob_subset.nc, Figure M1, unsmoothed 1° grid (67%)
python reversal_figures.py --data bob_subset.nc --out figures   # Figure M4 (seasonal_speed_drogued.png)
export BOB_DATA=$PWD/bob_subset.nc
bash report_code/run_all.sh               # every other figure and table (about an hour)
```

The scripts in `report_code/` are named after the report section they belong to
(for example `4_2_q2_coast_distance.py` makes Table Y and Figure W in Section 4.2).
**[`report_code/README.md`](report_code/README.md) has a table mapping every figure,
table and quoted number in the report to the script and output file that makes it.**
All random draws use seed 3001, so reruns give the same numbers; set `NB=20` for a
quick check with fewer redraws.

## What the report uses from `project3001.py`

`project3001.py` is the team's original end-to-end pipeline. The report uses three
things from it: the data download and Bay of Bengal subset (`bob_subset.nc`, which every
other script reads), the drifter-weighted 1° grid map (Figure M1, `06_circulation.png`)
and the unsmoothed 1° grid reversal share (67%, Section 4.3). It also writes many
exploratory figures that informed the project but are not in the final report; these
are kept so the full workflow can be followed, not because the report depends on them.

## Reuse

The study region, season months, smoothing distance, coverage rule (3 drifters within 100 km) and speed threshold (0.1 m/s) are set at the top of `reversal_figures.py`. Changing them maps another region or season without editing the functions.

## Limitations

Distances from the coast use a simplified hand-drawn coastline of India's east coast and Sri Lanka; white map areas are places no drifter visited, not a true land mask; and undrogued drifters (used only as a sensitivity check) are also pushed by the wind.
