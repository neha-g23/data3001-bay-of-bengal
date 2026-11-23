# Report figures and tables: Bay of Bengal monsoon reversal

Code for the figures, tables and numbers in our report that are not produced by
`project3001.py` or `reversal_figures.py` on their own. Every script uses the same
map as `reversal_figures.py`: drogued drifters, averaged within each drifter in each
1° cell, Gaussian kernel smoothing (75 km for seasons, 100 km for months) on a 0.2°
grid, shown only where at least 3 drifters lie within 100 km. Reversal = the current
turns more than 90° between the Southwest (Jun–Sep) and Northeast (Dec–Feb)
monsoons, counted only where the current is at least 0.1 m/s in both. Percentages
are area-weighted. 95% ranges come from redrawing whole drifters with replacement.

## Setup

```bash
pip install -r requirements.txt
```

Put `bob_subset.nc` in this folder (or set `BOB_DATA=/path/to/bob_subset.nc`).
Outputs are written to `report_outputs/`.

## Run

Run in this order (03 needs 01; 04 and 06 need 03):

```bash
python 01_headline_bootstrap.py
python 02_direction_change_map.py
python 03_region_table.py
python 04_region_map.py
python 05_monthly_switch.py
python 06_monthly_by_region.py
python 07_q2_coast_distance.py
python 08_q2_speed_turn.py
python 09_eda_figures.py
python 10_east_west_map.py
python 11_q2_speed_turn_figure.py
python 12_sensitivity_intervals.py
python 13_bandwidth_cv.py
python 14_kriging_diagram.py
python 15_monthly_turn_table.py
python 16_q2_coast_same_speed.py
python 17_table_e1.py
```

or `bash run_all.sh`. A full run takes roughly 30–60 minutes, mostly the bootstraps.
For a quick check, set fewer redraws: `NB=20 python 03_region_table.py`
(the ranges will then differ slightly from the report).

## What each script makes

| Script | Report item | Outputs |
|---|---|---|
| `01_headline_bootstrap.py` | 82% reverses (95% range 72–88%); 64% consistently reverses | `headline_bootstrap.csv`, `agreement_pct.npy`, `agreement_map.png` (Q3 figure) |
| `02_direction_change_map.py` | Q1 direction change map; 48.5% / 33.5% / 18% split | `direction_change_map.png` |
| `03_region_table.py` | Q1 region table (regions A–F) | `region_table.csv`, `regions.npy` |
| `04_region_map.py` | Q1 map with regions outlined | `region_map.png` |
| `05_monthly_switch.py` | Q1 Figure Z, monthly switch near coast vs offshore | `monthly_switch.png`, `monthly_turn.csv` |
| `06_monthly_by_region.py` | Appendix: monthly switch by region | `monthly_switch_by_region.png`, `monthly_switch_regions_combined.png`, `monthly_regions.csv` |
| `07_q2_coast_distance.py` | Q2 Table Y and Figure W (distance from the coast) | `q2_bands.csv`, `q2_coast_distance.png` |
| `08_q2_speed_turn.py` | Q2 speed effect: +18° per 0.1 m/s at the same distance | `speed_turn.csv` |
| `16_q2_coast_same_speed.py` | Q2 coast effect at similar speeds (about -11°) and speed effect (about 17° per 0.1 m/s), with and without offshore locations near other coasts | `coast_same_speed.csv` |
| `17_table_e1.py` | EDA Table E1 (cells kept per minimum drifters) and mean drifters per cell (10.8 / 5.6) | `table_e1.csv` |
| `12_sensitivity_intervals.py` | Q3 Table Z: sensitivity tests (smoothing, speed, cut-offs, years, all drifters, coverage rule), each with a 95% CI | `sensitivity_intervals.csv` |
| `15_monthly_turn_table.py` | Q2 Table V: median turn by month, near the coast and offshore | `monthly_completeness.csv` |
| `13_bandwidth_cv.py` | Methods Table M1: bandwidth selection by cross-validation (needs scikit-learn) | `bandwidth_cv.csv` |
| `11_q2_speed_turn_figure.py` | Q2 figure: median turn against speed, near the coast and offshore | `speed_vs_turn.png` |
| `09_eda_figures.py` | EDA Figures E1–E6 and Tables E1–E4 numbers | `eda_*.png`, `eda_numbers.json` |
| `10_east_west_map.py` | East-west current map, drogued drifters | `east_west_drogued.png` |

Other report results come from elsewhere in the repository:

- **Kernel smoothing vs kriging (Table W):** `../method_comparison/` (see its README)
- **Unsmoothed 1° grid result (67%, 55–74%)** and EDA Figures E7–E11: `project3001.py`

## Files

- `common.py`: data path, output folder, number of redraws (`NB`), shared map helpers
- `reversal_figures.py`: loads the main `../reversal_figures.py` (smoothing, coverage rule, direction change), so there is only one copy
- `reversal_uncertainty.py`: fast bootstrap weights and sensitivity tests; `python reversal_uncertainty.py --data bob_subset.nc --out report_outputs` also prints the arrow-stability numbers (median 5° and 10°) and the 9% uncertain area in Section 4.3
- `baseline_vs_kernel.py`: 1° grid vs kernel smoothing on held-out drifters (used by `../method_comparison/`)

## Notes

- Seed 3001 is used everywhere, so reruns with the same `NB` give the same numbers.
- Region boundaries are fixed lines (150 km from the coast, 10°N, 15°N, 88°E), and the
  coastline is a simplified hand-drawn line, so distances are approximate.
- Months or regions with few drifters give wide ranges; scripts 05–06 flag them.
