# PROJECT PROPOSAL
## 1. Research Question and Objective 
## 2. Region and Data

**Region R:** Bay of Bengal, **5°N–23°N, 78°E–99°E**.

* **Dataset**: NOAA GDP 1-hourly dataset (`gdp1h`, Elipot et al., 2016) accessed via `clouddrift`.
* **Variables**: `id` (drifter buoy ID), `time` (1-hourly timestamp), `latitude` (°N), `longitude` (°E), `ve` (eastward velocity, m/s), `vn` (northward velocity, m/s), `drogue_presence` (boolean sea-anchor status).

| Measure | Value | Acceptable range |
|---|---|---|
| Global hourly observations | 197,214,787 | – |
| Bay of Bengal observations | 866,511 (≈0.44% of global observations) | 0.8–3.9 million  |
| Distinct drifters in Bay of Bengal | 388 | 363–1,106  |
| Mean observations per drifter | ≈2,230 hours (≈93 days) | – |

**Data adequacy:** Both counts meet the brief's range, but it is only just above the minimums. Because hourly data from one drifter are strongly autocorrelated, the effective sample is closer to 388 drifters than 866,511 rows. We therefore focus on seasonal maps and flag cells with insufficient coverage rather than over-interpret them without success.

## 3.Significance of the Project

## 4.Intro Background and Existing Studies 

## 5.Proposed Method

## 6.Initial analysis and visualisation

## 7.Timeline Plan
| Phase / Milestone | Target Date | Key Deliverables & Focus |
| :--- | :--- | :--- |
| **Phase 1: Setup & Proposal** | Weeks 1–4 | Region R selection, data-adequacy validation, repository setup, and **Project Proposal / README** (15%, Due W4). |
| **Phase 2: Baseline Climatology** | Weeks 4–5 | Extract velocity vectors (`ve`, `vn`), build initial 1° × 1° mean circulation maps, and deliver **Poster Session** (20%, W5). |
| **Phase 3: Feedback & Decomposition** | Weeks 6–7 | Complete **Peer Review of Posters** (5%, W6), apply drogue-status filtering, and split velocity fields into Southwest/Northeast monsoon seasons. |
| **Phase 4: Uncertainty & Weighting** | Weeks 8–9 | Calculate cell-by-cell standard errors, spatial variance, and apply post-stratification to unweight drifter convergence traps. |
| **Phase 5: Presentation & Final Report** | Weeks 10–11 | Deliver **Group Presentation** (25%, W10), export standardized climatology dataset, and submit **Final Report** (40%, W11). |
