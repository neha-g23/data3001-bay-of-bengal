# Baseline vs kernel smoothing vs kriging

How well each mapping method predicts drifter velocities it has **not** seen. Bay of Bengal, NOAA Global Drifter Program (hourly, 2000–2022), drogued drifters only, Southwest (Jun–Sep) and Northeast (Dec–Feb) monsoons.

## Short answer

| Test | Best method | Kernel smoothing vs baseline | Kernel smoothing vs kriging |
|---|---|---|---|
| Hide whole years | **Kernel smoothing** | 5.7% lower error | Lower error overall; better in 5/5 year groups than kriging at cell centres, 4/5 than kriging at actual positions |
| Hide drifters | **Kriging (actual positions)** | 2.5% lower error | Kriging at actual positions has 3–5% lower error than kernel smoothing, in all 5 folds |

So the honest summary is: **kriging predicts slightly better when the held-out drifters come from years it has already seen; kernel smoothing predicts better for years it has never seen.** We use kernel smoothing because our maps are meant to describe the typical monsoon, not one particular year, and it is the method that generalises best to new years.

## What the numbers mean

- **RMSE (m/s), "vector RMSE"**: for each held-out point, the distance between the predicted velocity arrow and the drifter's real velocity arrow, sqrt((predicted east − actual east)² + (predicted north − actual north)²). Square these, average over all held-out points, take the square root. Lower is better. About 0.36–0.42 m/s here, which is large because single drifters include eddies and wind wobble no smooth map can predict.
- **% lower than baseline** = (1 − method RMSE ÷ baseline RMSE) × 100. Positive = better than the baseline.
- **n_points**: number of held-out points scored. A point = one drifter's average velocity in one 1° square in one monsoon (and, in the year test, one year). Only points that **every** method can predict are scored, so all methods are compared on exactly the same points.
- **95% range** (where given): from resampling whole drifters (drifter test) or whole years (year test) 2,000–4,000 times.

## The methods

| Method | What it does |
|---|---|
| **Baseline (1° grid)** | Average of the training drifters in the same 1° square (needs ≥ 3 drifters). |
| **Kernel smoothing (75 km)** | Weighted average of nearby training drifters; weight falls off as a Gaussian with 75 km bandwidth. Needs ≥ 3 drifters within 100 km. This is the method used for all our maps. |
| **Kriging, actual positions** | Gaussian-process regression (scikit-learn) on each drifter's actual average position, separately for east and north velocity and for each monsoon. Two covariance shapes tried: Matérn 3/2 and exponential (Matérn 1/2), with a noise term. |
| **Kriging, cell centres** | Same, but each point is placed at the centre of its 1° square instead of its actual position. |

## The two tests

1. **Hide drifters** (`scripts/kpos.py`, stats in `scripts/kpos_stats.py`): drifters split into 5 random groups; each group is hidden in turn, the map is built from the other 4 and used to predict the hidden drifters. 2,841 points from 260 drifters. Same folds and points as `baseline_vs_kernel.py`, so the 2.5% matches the presentation.
2. **Hide whole years** (`scripts/kyear.py`, stats in `scripts/kyear_stats.py`): monsoon-years split into 5 groups of similar size (2019 alone, because it is the biggest year). Each group is hidden in turn. 2,492 points from 256 drifters. This is a harder test: no drifter from the hidden years helps.

## Results

### Hide drifters (5 folds)

| Method | RMSE (m/s) | % lower than baseline |
|---|---|---|
| Baseline (1° grid) | 0.3655 | 0.0 |
| Kernel smoothing (75 km) | 0.3563 | 2.5 (95% range 1.3–3.9) |
| Kriging, Matérn 3/2, actual positions | 0.3453 | 5.5 |
| Kriging, exponential, actual positions | **0.3401** | **6.9** |
| Kriging, Matérn 3/2, cell centres | 0.3584 | 1.9 |
| Kriging, exponential, cell centres | 0.3581 | 2.0 |

### Hide whole years (5 groups)

| Method | RMSE (m/s) | % lower than baseline |
|---|---|---|
| Baseline (1° grid) | 0.4244 | 0.0 |
| Kernel smoothing (75 km) | **0.4001** | **5.7** (95% range 3.8–9.2) |
| Kriging, Matérn 3/2, cell centres | 0.4086 | 3.7 |
| Kriging, exponential, actual positions | 0.4120 | 2.9 |

Kernel smoothing vs kriging in the year test: 2.1% lower error than Matérn kriging (95% range 1.1–3.5%) and 3.0% lower than exponential kriging (95% range 0.0–7.5%, so only just clear of zero).

### Hide whole years, per year group (RMSE, m/s)

| Years hidden | Baseline | Kernel | Kriging (Matérn, centres) | Kriging (exp., actual) |
|---|---|---|---|---|
| 2019 | 0.427 | **0.416** | 0.417 | 0.419 |
| 2004, 2006, 2015, 2017, 2021 | 0.373 | 0.357 | 0.368 | **0.344** |
| 2010, 2011, 2014 | 0.453 | **0.415** | 0.430 | 0.439 |
| 2005, 2007, 2008, 2009, 2013 | 0.461 | **0.441** | 0.445 | 0.454 |
| 2003, 2012, 2016, 2018, 2020 | 0.416 | **0.375** | 0.389 | 0.409 |

Full tables: `results_summary.csv`, `results_by_year_group.csv`.

## Why the two tests disagree (our interpretation)

When only some drifters are hidden, other drifters from the **same year** are still in the training data, often in the same eddies. Kriging fits that fine, year-specific structure closely, so it wins. When whole years are hidden, that structure is gone, and the simpler, smoother kernel map generalises better. This is our reading of the pattern, not something we tested directly.

## How to run

Needs `bob_subset.nc` and the main repo files (`reversal_figures.py`, `baseline_vs_kernel.py`) on the path, plus scikit-learn.

```bash
python scripts/kpos.py 1.5      # hide drifters, Matérn 3/2 kriging  -> kpos_1.5.pkl
python scripts/kpos.py 0.5      # hide drifters, exponential kriging -> kpos_0.5.pkl
python scripts/kpos_stats.py 1.5
python scripts/kyear.py         # hide whole years                   -> kyear.pkl
python scripts/kyear_stats.py
```

`kpos.py` reads `seasonal_pts.pkl`, the per-drifter seasonal points made by `baseline_vs_kernel.seasonal_points()`. Kriging is slow (several minutes per run).
