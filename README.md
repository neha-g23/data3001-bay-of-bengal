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
