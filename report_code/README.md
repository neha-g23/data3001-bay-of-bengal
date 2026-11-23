# Report figures and tables

Every script here is named after the **report section** it belongs to
(`4_2_q2_...` = Section 4.2, Q2). Each one writes its figure or table to
`report_outputs/` and prints the numbers quoted in the report.

All scripts use the same map as `../reversal_figures.py`: drogued drifters, averaged
within each drifter in each 1° cell, Gaussian kernel smoothing (75 km for seasons,
100 km for months) on a 0.2° grid, shown only where at least 3 drifters lie within
100 km. Reversal = the current turns more than 90° between the Southwest (Jun–Sep)
and Northeast (Dec–Feb) monsoons, counted only where the current is at least 0.1 m/s
in both. Percentages are area-weighted. 95% ranges come from redrawing whole drifters
with replacement (seed 3001).

## Setup and run

```bash
pip install -r ../requirements.txt
export BOB_DATA=/path/to/bob_subset.nc     # made by `python ../project3001.py`
bash run_all.sh                            # about an hour, mostly bootstraps
NB=20 python 4_1_q1_headline_bootstrap.py  # quick check with fewer redraws
```

`4_1_q1_region_table.py` needs `4_1_q1_headline_bootstrap.py` to have run first;
`4_1_q1_region_map.py` and `4_2_q2_monthly_turn_table.py` reuse the regions from it.
`run_all.sh` already runs them in the right order.

## Where each report item comes from

| Section | Report item | Script | Output |
|---|---|---|---|
| 3.2–3.3 | Data and EDA numbers (records, removals, drogued share, speeds, days per drifter) | `3_3_eda_numbers_and_figure.py` | `eda_numbers.json` |
| 3.3 | **Figure E1** (records per year, drifters per month, speeds, days per drifter) | `3_3_eda_numbers_and_figure.py` | `figure_e1_eda.png` |
| 3.3 | **Table E1** (cells kept per minimum drifters; 10.8 and 5.6 drifters per cell) | `3_3_eda_cell_counts_table.py` | `table_e1.csv` |
| 3.4.1 | **Figure M1** (drifter-weighted 1° grid) | `../project3001.py` | `outputs/.../06_circulation.png` |
| 3.4.2 | **Figure M2** (drifters per 0.2° square) | `3_4_drifter_density_map.py` | `figure_m2_drifter_density.png` |
| 3.4.3 | **Figure M3** (kernel smoothing vs kriging diagram) | `3_4_kernel_vs_kriging_diagram.py` | `figure_m3_kernel_vs_kriging.png` |
| 3.4.4 | **Figure M4** (kernel-smoothed seasonal currents) | `../reversal_figures.py` | `seasonal_speed_drogued.png` |
| 3.4.5 | **Table M1** (bandwidth cross-validation) | `3_4_bandwidth_cv_table.py` | `bandwidth_cv.csv` |
| 4.1 | Headline 82% reverses (95% CI 72–88%) | `4_1_q1_headline_bootstrap.py` | `headline_bootstrap.csv` |
| 4.1 | **Figure R** (direction change, four classes) | `4_1_q1_direction_change_classes.py` (same map as `../reversal_figures.py`) | `direction_change_map.png` |
| 4.1 | **Table X** (reversal by region A–F) | `4_1_q1_region_table.py` | `region_table.csv` |
| 4.1 | **Figure X** (regions map) | `4_1_q1_region_map.py` | `region_map.png` |
| 4.2 | **Table Y**, **Figure W** (reversal by distance from the coast) | `4_2_q2_coast_distance.py` | `q2_bands.csv`, `q2_coast_distance.png` |
| 4.2 | Speed effect: +18° per 0.1 m/s at the same distance | `4_2_q2_speed_turn.py` | `speed_turn.csv` |
| 4.2 | **Figure S** (turn against speed, coast vs offshore) | `4_2_q2_speed_turn_figure.py` | `speed_vs_turn.png` |
| 4.2 | Coast effect at the same speed (−11°, 95% CI −35 to +13) | `4_2_q2_coast_same_speed.py` | `coast_same_speed.csv` |
| 4.2 | **Figure Z** (monthly switch, coast vs offshore) | `4_2_q2_monthly_switch.py` | `monthly_switch.png`, `monthly_turn.csv` |
| 4.2 | **Table V** (median turn by month) | `4_2_q2_monthly_turn_table.py` | `monthly_completeness.csv` |
| 4.3 | **Figure U** (share of redraws in which each location reverses), 64% consistently reverses | `4_1_q1_headline_bootstrap.py` | `agreement_map.png` |
| 4.3 | **Table Z** (sensitivity tests) | `4_3_q3_sensitivity_table.py` | `sensitivity_intervals.csv` |
| 4.3 | Without smoothing: 67% of 1° cells reverse (95% CI 55–74%) | `../project3001.py` | printed, and `09_reversal_map.png` |
| 4.3 | **Table W** (grid vs kernel vs kriging on held-out data) | `../method_comparison/` (see its README) | `../method_comparison/*.csv` |

## Shared code

| File | What it holds |
|---|---|
| `common.py` | Data loading, output folder, seed, coastline distance, plotting helpers |
| `reversal_uncertainty.py` | Whole-drifter bootstrap and the sensitivity variants |
| `baseline_vs_kernel.py` | Shared fitting code for the held-out comparison in `../method_comparison/` (Table W) |
| `reversal_figures.py` | Small loader for `../reversal_figures.py`, so the scripts can import it |
