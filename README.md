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


## How to Reproduce the Analysis

### What you need

- **Python 3.13** (other recent versions may work, but 3.13 is what was tested).
  Download it from https://www.python.org/downloads/.
- **The files in this folder:** `project3001.py`, `pyproject.toml`, `uv.lock`,
  `requirements.txt` and, ideally, `bob_subset.nc` (the Bay of Bengal data,
  about 70 MB).
- **Not** the `.venv313` or `.venv` folders. These only work on the computer
  they were created on, so set up your own (below).

### Setup (once)

Open a terminal in this folder and set up an environment with the required
packages, using **one** of the two options below.

#### Option A: uv (recommended, exact reproduction)

[uv](https://docs.astral.sh/uv/getting-started/installation/) reads
`pyproject.toml` and `uv.lock`, installs Python 3.13 if needed, and installs
exactly the package versions the analysis was tested with (including every
indirect dependency).

```
uv sync
```

Then run the analysis with:

```
uv run python project3001.py
```

#### Option B: pip

**Windows**

```
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

**Mac / Linux**

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Run

If you used uv, run `uv run python project3001.py` (see above). With pip:

**Windows**

```
.venv\Scripts\python project3001.py
```

**Mac / Linux**

```
.venv/bin/python project3001.py
```

With `bob_subset.nc` in the folder, a run takes about a minute.

Without `bob_subset.nc`, the first run downloads the global drifter dataset
over the internet and cuts it down to the Bay of Bengal. This takes several
minutes and needs a good connection; the result is saved as `bob_subset.nc`
so later runs are fast. If you can, get a copy of `bob_subset.nc` instead
(it is often too large for email; use a shared drive or cloud link).

The first run also downloads coastline data (about 2 MB) to draw land on the
maps. If that download fails, the maps are drawn without land and all results
are unaffected.

### Outputs

Everything is written to `outputs/`:

| Path | Contents |
| --- | --- |
| `figures/` | maps and charts (PNG) |
| `tables/` | gridded climatology and supporting tables (CSV) |
| `report.txt` | full printed log of the run |
| `summary.txt` | headline numbers |
| `reversal_caption.txt` | ready-to-use caption for the reversal map |

### Troubleshooting

- **`No Python at ...` or `did not find executable`**: you are using a copied
  `.venv313` folder. Delete it and follow the setup steps above.
- **`ModuleNotFoundError: No module named ...`**: the packages are not
  installed in the Python you are running. Run the setup steps, then start the
  script with the `.venv` Python as shown under *Run*.
- **Very slow first run or a download error**: `bob_subset.nc` is missing; see
  above.
