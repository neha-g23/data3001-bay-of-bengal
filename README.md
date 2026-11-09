# Bay of Bengal monsoon reversal figures

`reversal_figures.py` makes every reversal figure for our DATA3001 project from the cached Global Drifter Program subset (`bob_subset.nc`, hourly drifter data, 14 July 2000 to 30 June 2022, 5–23°N, 78–99°E).

## How to run

```bash
pip install -r requirements.txt
python reversal_figures.py --data bob_subset.nc --out figures
```

Options:

- `--drifters drogued`: main figures only (drogued drifters)
- `--drifters all`: robustness versions only (drogued + undrogued)
- `--drifters both`: both sets (default)

The whole run takes about a minute. `bob_subset.nc` is not in the repository; put your copy next to the script.

## What it produces

Each figure is made twice, ending in `_drogued.png` (main) or `_all_drifters.png` (robustness check).

| File | What it shows |
|---|---|
| `seasonal_speed_*` | Smoothed surface currents in the Southwest and Northeast monsoons (colour = speed) |
| `seasonal_east_west_*` | Same, coloured by east-west flow (red = east, blue = west) |
| `direction_change_map_*` | Change in current direction between the two monsoons (0–180°) |
| `direction_change_map_annotated_*` | Same, with the share reversing in each latitude band and the coastal band |
| `direction_change_chart_*` | Distribution of direction change, and share reversing by region |
| `reversal_vs_speed_*` | Share reversing against current speed |
| `reversal_vs_coast_distance_*` | Speed and reversal against distance from India's east coast and Sri Lanka |
| `monthly_speed_*` | Smoothed currents for each monsoon month (June–September, December–February) |
| `monthly_east_west_*` | East-west flow for all 12 months |
| `monthly_reversal_maps_*`, `monthly_reversal_line_*` | Direction change of each month relative to July |
| `eicc_timeline_*` | Monthly along-coast current within 150 km of India's east coast, with bootstrap intervals |

The script also prints the key numbers (reversal shares, distance-band table, monthly shares, coastal velocities).

## Method

1. **Cleaning:** drop missing values and speeds above 3 m/s.
2. **Seasons:** Southwest monsoon = June–September, Northeast monsoon = December–February.
3. **Drifter weighting:** average within each drifter in each 1° cell (at least 12 hourly observations), so a drifter that lingers cannot dominate.
4. **Kernel smoothing:** Gaussian kernel on great-circle distance, 75 km for seasonal maps and 100 km for monthly maps. Both bandwidths were chosen by cross-validation that holds out whole drifters (in the main project code).
5. **Coverage rule:** a location is shown only if at least 3 different drifters lie within 100 km. This radius is fixed, so maps at different bandwidths differ only in smoothness.
6. **Direction change:** angle between the Southwest and Northeast current at each location. Currents slower than 0.1 m/s are excluded from reversal percentages because their direction is not meaningful.
7. **Area weighting:** percentages are weighted by cos(latitude), because grid cells shrink towards the north.

## Settings

All settings are at the top of the script (`BANDWIDTH_SEASON_KM`, `MIN_DRIFTERS`, `SUPPORT_KM`, `MIN_SPEED_DIRECTION`, and so on). Change them there rather than in the functions.

## Limitations

- The coastline used for distances and the coastal band is a simple hand-drawn line, so distances are approximate.
- "White" areas are places no drifter has visited (mostly land), not a true land mask.
- Undrogued drifters are also pushed by the wind, so the `_all_drifters` versions may slightly overstate current speeds.
- Smoothed reversal percentages are higher than the gridded estimate in the main analysis, because smoothing removes noise and slow currents are excluded. State which definition you use.
