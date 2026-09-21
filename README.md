# PROJECT PROPOSAL
## 1. Research Question and Objective 
## 2. Region and Data

**Region R:** Bay of Bengal, **5°N–23°N, 78°E–99°E** (excludes the equator).

* **Dataset**: NOAA GDP 1-hourly dataset (`gdp1h`, Elipot et al., 2016) accessed via `clouddrift`.
* **Variables**: `id` (drifter buoy ID), `time` (1-hourly timestamp), `latitude` (°N), `longitude` (°E), `ve` (eastward velocity, m/s), `vn` (northward velocity, m/s), `drogue_presence` (boolean sea-anchor status).

| Measure | Value | Brief's acceptable range |
|---|---|---|
| Global hourly observations | 197,214,787 | – |
| Bay of Bengal observations | 866,511 (≈0.44% of global observations) | 0.8–3.9 million  |
| Distinct drifters in Bay of Bengal | 388 | 363–1,106  |
| Mean observations per drifter | ≈2,230 hours (≈93 days) | – |

**Data adequacy:** Both counts meet the brief's range, but it is only just above the minimums. Because hourly data from one drifter are strongly autocorrelated, the effective sample is closer to 388 drifters than 866,511 rows. We therefore focus on seasonal maps and flag cells with insufficient coverage rather than over-interpret them without success.

## Significance of the Project

## Intro Background and Existing Studies 

## Proposed Method

## Initial analysis and visualisation

## Timeline Plan

