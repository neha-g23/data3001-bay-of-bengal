r"""
Bay of Bengal Surface Circulation: Southwest vs Northeast Monsoon
=================================================================

Purpose
-------
Estimates how strongly the surface currents of the Bay of Bengal reverse
between the Southwest monsoon (June-September) and the Northeast monsoon
(December-February), and how confident we can be in those estimates. The
result is a gridded seasonal climatology of mean surface current and sea
surface temperature (SST), with an uncertainty estimate for every grid cell.

Data
----
NOAA Global Drifter Program (GDP) hourly drifter dataset, accessed through the
`clouddrift` package, restricted to 5-23°N, 78-99°E.

Method in brief
---------------
1. Clean the data: remove observations with missing time, position or
   velocity, speeds above 3 m/s, and duplicate records.
2. Choose the dataset: drogued drifters (the default, since undrogued drifters
   are pushed by the wind), undrogued drifters, or all drifters. Every step
   below runs the same way on whichever dataset is chosen, and a final
   comparison step sets the three results side by side.
3. Grid the observations into 1° cells. A cell is mapped only if at least
   3 different drifters each spent at least 12 hours in it.
4. Average each drifter's mean velocity in a cell, so that every drifter counts
   equally. This stops a single drifter caught in an eddy from dominating.
5. Estimate uncertainty by resampling whole drifters (bootstrap). Cells with
   10 or more drifters are labelled high confidence, 3-9 low confidence.
6. Report the percentage of cells where the current direction changes by more
   than 90° between monsoons, for the whole Bay and for the western Bay
   (80-85°E, an approximate East India Coastal Current band), with 95%
   confidence intervals. To show how much of this is caused by the monsoon,
   the same comparison is made between different years of the same season,
   which shows how much direction change occurs without a monsoon contrast.
7. As an alternative to the fixed seasons, build a map for every calendar
   month in which observations from neighbouring months count with smoothly
   decreasing weight. The amount of smoothing is chosen by cross-validation
   and compared with the fixed seasons.

How to run
----------
    .venv313\Scripts\python.exe project3001.py                        # default dataset (drogued)
    .venv313\Scripts\python.exe project3001.py --dataset undrogued    # or: drogued, all
    .venv313\Scripts\python.exe project3001.py --dataset every        # all three, then the comparison
    .venv313\Scripts\python.exe project3001.py --compare              # comparison only, from saved results

The first run downloads and subsets the global dataset (a few minutes) and
caches it to `bob_subset.nc`. Later runs take about a minute per dataset.

Outputs (written to `outputs/<dataset>/`, e.g. `outputs/drogued/`)
-----------------------------------------------------------------
    figures/   maps and charts (PNG)
    tables/    gridded climatology and supporting tables (CSV)
    report.txt full printed log of the run
    summary.txt headline numbers
    reversal_caption.txt  ready-to-use caption for the reversal map

The comparison of the three datasets is written to `outputs/comparison/`.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")  # draw straight to files, never open a window
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import xarray as xr
from scipy.linalg import cho_solve, cholesky, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


# ============================================================
# Settings
# ============================================================
# All method choices are set here so they can be reviewed or changed in one
# place. The rest of the script reads these values and contains no other
# hard-coded thresholds.

# --- Study region and grid ---
LON_RANGE = (78.0, 99.0)         # longitude limits, degrees East
LAT_RANGE = (5.0, 23.0)          # latitude limits, degrees North
GRID = 1.0                       # grid cell size, degrees

# --- Monsoon seasons (calendar month -> season; other months are not analysed) ---
SEASONS = ["Southwest", "Northeast"]
SEASON_OF_MONTH = {6: "Southwest", 7: "Southwest", 8: "Southwest", 9: "Southwest",
                   12: "Northeast", 1: "Northeast", 2: "Northeast"}

# --- Quality control and mapping rules ---
MAX_SPEED = 3.0                  # m/s; faster observations are removed as unrealistic
MIN_OBS_PER_DRIFTER = 12         # hours a drifter must spend in a cell to count towards it
MIN_DRIFTERS = 3                 # qualifying drifters a cell needs before it is mapped
HIGH_CONF_DRIFTERS = 10          # cells with at least this many drifters are "high confidence"
ERR_SST_MAX = 0.5                # °C; SST readings with a larger reported error are removed
MIN_SPEED_FOR_PCT = 0.05         # m/s; percentage speed differences are only shown above this,
                                 # since a percentage of a near-zero speed is not meaningful

# --- Uncertainty estimation ---
N_BOOT = 2000                    # number of bootstrap resamples (enough for stable 95% intervals:
                                 # each 2.5% tail then rests on 50 resamples)
SEED = 0                         # random seed, so results are reproducible

# --- East India Coastal Current (EICC) band ---
EICC_LON = (80.0, 85.0)          # western Bay longitude band used as an approximate EICC region
                                 # (a simple longitude band, not a coastline-following mask)

# --- Smooth temporal weighting (alternative to the fixed seasons above) ---
TEMPORAL_SIGMAS = (0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)   # widths of the Gaussian month weighting
                                                          # to test, in months; 0 = target month only

# --- Interannual baseline (same season, different years) ---
BASELINE_MIN_DRIFTERS = 1        # drifters a cell needs in each half-data map. Lower than MIN_DRIFTERS
                                 # because each map has only half the data; the extra noise affects
                                 # both comparisons equally, so the comparison stays fair
BASELINE_THRESHOLDS = (135, 90)  # direction changes (degrees) tested; > 135° (near-opposite flow)
                                 # is the main definition of a reversal for this test
BASELINE_SENS_CELLS = [1, 2, 3]  # cell sizes (degrees) and drifter minimums tried in the
BASELINE_SENS_MIN = [1, 2, 3]    # sensitivity test of the baseline

# --- Threshold sensitivity test: combinations of mapping rules to compare ---
SENS_MIN_DRIFTERS = [1, 3, 5, 10]
SENS_MIN_OBS = [1, 6, 12, 24, 48]

# --- Dataset ---
# Which drifters the analysis uses. Every step reads the chosen dataset, so
# switching datasets needs no other change. The default can be overridden on
# the command line with --dataset (see "How to run" above).
DATASETS = {
    "drogued": "drogued drifters only",      # follow the water; the default and recommended choice
    "undrogued": "undrogued drifters only",  # drogue lost, so also pushed by the wind
    "all": "all drifters",                   # drogued and undrogued together
}
DATASET = "drogued"

# --- File locations ---
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "bob_subset.nc")   # local copy of the regional data;
                                                   # delete it if the region above changes
OUTPUT_ROOT = os.path.join(HERE, "outputs")        # each dataset gets its own folder inside this
COMPARISON = "comparison"                          # folder for the three-dataset comparison

# Output folders for the dataset currently being analysed (set by set_dataset)
ACTIVE = DATASET
OUT_DIR = os.path.join(OUTPUT_ROOT, ACTIVE)
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

# Columns that identify one grid cell in one season. lon0 / lat0 are the
# south-west corner of the cell; lon / lat (added later) are its centre.
CELL = ["monsoon", "lon0", "lat0"]

plt.rcParams.update({"font.size": 12})
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


# ============================================================
# Output helpers
# ============================================================
# Small utilities for writing the report, figures and tables to outputs/.

class Tee:
    """Send printed output to the screen and to report.txt at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def set_dataset(name):
    """Make `name` the active dataset (or the comparison) and point all
    outputs at its own folder, outputs/<name>/, creating it if needed."""
    global ACTIVE, OUT_DIR, FIG_DIR, TAB_DIR
    ACTIVE = name
    OUT_DIR = os.path.join(OUTPUT_ROOT, name)
    FIG_DIR = os.path.join(OUT_DIR, "figures")
    TAB_DIR = os.path.join(OUT_DIR, "tables")
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)


def dataset_label():
    """Readable name of the active dataset, used in titles and headings."""
    return DATASETS.get(ACTIVE, ACTIVE)


def select_dataset(obs, name):
    """The observations belonging to a dataset: drogued, undrogued or all."""
    if name == "drogued":
        return obs[obs["drogue"]]
    if name == "undrogued":
        return obs[~obs["drogue"]]
    return obs


def section(title):
    """Print a section heading in the report."""
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def save_fig(fig, name, dpi=200):
    """Save a figure to outputs/figures/ and close it."""
    path = os.path.join(FIG_DIR, name)
    if os.path.exists(path):
        os.remove(path)     # remove the old file first; overwriting in place can fail in
                            # cloud-synced folders such as OneDrive
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> figures/{name}")


def save_table(df, name, index=False):
    """Save a table to outputs/tables/ as CSV."""
    df.to_csv(os.path.join(TAB_DIR, name), index=index)
    print(f"  -> tables/{name}")


# ============================================================
# Land overlay (Natural Earth coastline)
# ============================================================
# Land is drawn as semi-transparent coastline shapes on top of the coloured
# data cells, so coastal cells remain visible underneath, and below the current
# arrows. The coastline comes from Natural Earth (1:50m scale, about 2 MB) via
# cartopy and is downloaded once. If it cannot be loaded, a warning is printed
# and the maps are drawn without land; all results are unaffected.

LAND_COLOR = (0.78, 0.71, 0.54, 0.35)   # semi-transparent sand colour, chosen to be
                                        # distinct from the grey "insufficient data" hatching
COAST_COLOR = "#4d4d4d"
LAND_ZORDER = 1.5           # drawing order: data cells (0) < land (1.5) < arrows (3)

_land_polys = None   # coastline shapes, loaded on first use ([] if loading failed)


def _polygons_of(geom):
    """Break a shape into its individual polygons."""
    if geom.geom_type == "Polygon":
        yield geom
    elif hasattr(geom, "geoms"):
        for g in geom.geoms:
            yield from _polygons_of(g)


def _get_land(margin=1.0):
    """Coastline polygons clipped to the study region (loaded once, then reused)."""
    global _land_polys
    if _land_polys is None:
        try:
            import cartopy.io.shapereader as shpreader
            from shapely.geometry import box
            clip = box(LON_RANGE[0] - margin, LAT_RANGE[0] - margin,
                       LON_RANGE[1] + margin, LAT_RANGE[1] + margin)
            path = shpreader.natural_earth(resolution="50m", category="physical", name="land")
            _land_polys = [
                np.asarray(poly.exterior.coords)
                for geom in shpreader.Reader(path).geometries()
                for poly in _polygons_of(geom.intersection(clip))
                if not poly.is_empty
            ]
        except Exception as err:
            _land_polys = []
            print(f"\n[land] Land NOT drawn ({type(err).__name__}: {err}). "
                  "Check cartopy is installed and you are online for the one-off download.\n")
    return _land_polys


def add_land(ax):
    """Draw land on a map."""
    if _get_land():
        ax.add_collection(PolyCollection(_get_land(), facecolors=LAND_COLOR, edgecolors=COAST_COLOR,
                                         linewidths=0.6, zorder=LAND_ZORDER), autolim=False)


EICC_LINE_STYLE = dict(color="black", linestyle="--", linewidth=1.2, zorder=4)


def add_land_legend(ax, extra_handles=(), loc="upper left", fontsize=9):
    """Map legend showing land, the EICC band, and any extra entries passed in."""
    handles = list(extra_handles)
    if _get_land():
        handles.append(Patch(facecolor=LAND_COLOR, edgecolor=COAST_COLOR, label="Land"))
    handles.append(plt.Line2D([], [], **EICC_LINE_STYLE,
                              label=f"Approx. EICC band ({EICC_LON[0]:.0f}-{EICC_LON[1]:.0f}°E)"))
    ax.legend(handles=handles, loc=loc, fontsize=fontsize, framealpha=0.95)


# ============================================================
# Map helpers
# ============================================================
# Every map in the report is drawn with these functions, so all maps share the
# same layout, colour handling and "insufficient data" hatching.

INSUFFICIENT_STYLE = dict(facecolor="lightgrey", edgecolor="grey", hatch="//", alpha=0.6, linewidth=0)
INSUFFICIENT_PATCH = Patch(**INSUFFICIENT_STYLE,
                           label=f"Insufficient data (<{MIN_DRIFTERS} drifters with "
                                 f">={MIN_OBS_PER_DRIFTER} obs)")


def _cell_polys(g):
    """The four corners of each grid cell, for drawing."""
    return [[(x, y), (x + GRID, y), (x + GRID, y + GRID), (x, y + GRID)]
            for x, y in zip(g["lon0"], g["lat0"])]


def setup_map(ax, title, xlim=LON_RANGE):
    """Axis limits, labels, title, and dashed lines marking the EICC band."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*LAT_RANGE)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    for lon in EICC_LON:
        ax.axvline(lon, **EICC_LINE_STYLE)


def draw_cells(ax, g, values, cmap, norm):
    """Fill each grid cell with a colour representing its value."""
    if len(g):
        ax.add_collection(PolyCollection(_cell_polys(g), array=np.asarray(values, dtype=float),
                                         cmap=cmap, norm=norm, edgecolors="none", zorder=0),
                          autolim=False)


def draw_insufficient(ax, g):
    """Hatch cells that were sampled but have too few drifters to be mapped."""
    if len(g):
        ax.add_collection(PolyCollection(
            _cell_polys(g), facecolors=INSUFFICIENT_STYLE["facecolor"],
            edgecolors=INSUFFICIENT_STYLE["edgecolor"], hatch="//",
            alpha=INSUFFICIENT_STYLE["alpha"], linewidths=0, zorder=0), autolim=False)


def season_cell_map(ax, g, value_col, cmap, norm, title, arrows=False):
    """One season's map: cells with enough data are coloured by value_col, cells
    without enough data are hatched, and arrows (optional) show current direction.
    Arrows all have the same length; speed is shown by the cell colour."""
    ok = g[g["sufficient"]]
    draw_insufficient(ax, g[~g["sufficient"]])
    draw_cells(ax, ok, ok[value_col], cmap, norm)
    if arrows and len(ok):
        speed = np.where(ok["speed"] > 0, ok["speed"], 1.0)
        ax.quiver(ok["lon"], ok["lat"], ok["ve"] / speed, ok["vn"] / speed,
                  color="black", angles="xy", scale=25, width=0.003, zorder=3)
    add_land(ax)
    setup_map(ax, title)


def add_colorbar(fig, axes, cmap, norm, label):
    """Shared colour scale for one or more maps."""
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                        location="right", shrink=0.85, pad=0.03)
    cbar.set_label(label)
    return cbar


# ============================================================
# Data loading and cleaning
# ============================================================

def load_subset():
    """Load the Bay of Bengal drifter data. Uses the local copy if one exists;
    otherwise streams the global GDP hourly dataset, cuts it to the study
    region, and saves a local copy for future runs."""
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached subset: {CACHE_PATH}")
        return xr.open_dataset(CACHE_PATH, engine="h5netcdf")

    import clouddrift as cd
    from clouddrift.ragged import subset
    print("Computing subset from gdp1h() (this takes a few minutes)...")
    ds = subset(cd.datasets.gdp1h(), {"lat": LAT_RANGE, "lon": LON_RANGE}, row_dim_name="traj")
    ds.load()
    ds.to_netcdf(CACHE_PATH, engine="h5netcdf")
    print("Saved subset to disk.")
    return ds


def data_quality_table(ds):
    """Missing-value counts and value ranges for the key variables, before cleaning."""
    cols = ["lat", "lon", "ve", "vn", "err_ve", "err_vn", "sst", "err_sst", "drogue_status", "gap"]
    df = ds[[c for c in cols if c in ds]].to_dataframe()
    numeric = df.select_dtypes("number")
    return pd.DataFrame({
        "missing": df.isna().sum(),
        "missing_pct": df.isna().mean() * 100,
        "min": numeric.min(),
        "mean": numeric.mean(),
        "max": numeric.max(),
    })


def build_obs_table(ds):
    """Flatten the dataset into a table with one row per hourly observation,
    labelled with the drifter that made it. The source data stores each
    drifter's observations one after another, along with how many there are,
    so repeating each drifter ID that many times labels every observation."""
    return pd.DataFrame({
        "drifter_id": np.repeat(ds["id"].values, ds["rowsize"].values.astype(int)),
        "time": pd.to_datetime(ds["time"].values),
        "lat": ds["lat"].values,
        "lon": ds["lon"].values,
        "ve": ds["ve"].values,
        "vn": ds["vn"].values,
        "sst": ds["sst"].values - 273.15,   # stored in Kelvin
        "err_sst": ds["err_sst"].values,
        "drogue": ds["drogue_status"].values.astype(bool),
    })


def clean_obs(obs):
    """Remove missing, unrealistic and duplicate observations, recording how many
    each step removed. Then label each observation with its monsoon season and
    the grid cell it falls in."""
    steps = [{"step": "raw observations", "removed": 0, "remaining": len(obs)}]

    def drop(mask, reason):
        nonlocal obs
        obs = obs[~np.asarray(mask)]
        steps.append({"step": reason, "removed": int(np.sum(mask)), "remaining": len(obs)})

    drop(obs[["time", "lat", "lon", "ve", "vn"]].isna().any(axis=1),
         "missing time / position / velocity")
    drop(np.hypot(obs["ve"], obs["vn"]) > MAX_SPEED, f"speed > {MAX_SPEED} m/s")
    drop(obs.duplicated(["drifter_id", "time"]), "duplicate drifter + time")

    obs = obs.copy()
    obs["month"] = obs["time"].dt.month
    obs["year"] = obs["time"].dt.year
    obs["monsoon"] = obs["month"].map(SEASON_OF_MONTH).fillna("Inter-monsoon")

    # Grid cell (south-west corner) of each observation. Points exactly on the
    # outer edge of the region are placed in the last cell rather than outside it.
    n_cells_x = int(round((LON_RANGE[1] - LON_RANGE[0]) / GRID))
    n_cells_y = int(round((LAT_RANGE[1] - LAT_RANGE[0]) / GRID))
    obs["lon0"] = LON_RANGE[0] + GRID * np.clip(np.floor((obs["lon"] - LON_RANGE[0]) / GRID), 0, n_cells_x - 1)
    obs["lat0"] = LAT_RANGE[0] + GRID * np.clip(np.floor((obs["lat"] - LAT_RANGE[0]) / GRID), 0, n_cells_y - 1)
    return obs, pd.DataFrame(steps)


# ============================================================
# Gridding: drifter-weighted cell means
# ============================================================
# A drifter caught in an eddy can record hundreds of hours in one cell. If every
# hourly observation counted equally, that one drifter would dominate the cell.
# Instead, each drifter's observations in a cell are first averaged, and the
# cell value is the average of those drifter means: one drifter, one vote.

def per_drifter_means(obs, cols=("ve", "vn")):
    """Mean of each variable for every drifter, in every cell, in each monsoon,
    together with how many hourly observations that drifter contributed."""
    named = {c: (c, "mean") for c in cols}
    return (obs.groupby(CELL + ["drifter_id"])
               .agg(**named, n_obs=(cols[0], "count"))
               .reset_index())


def cell_grid(per, cols, min_obs=MIN_OBS_PER_DRIFTER, min_drifters=MIN_DRIFTERS):
    """Cell value = average of the per-drifter means, so each drifter counts once.
    Only drifters with at least min_obs hourly observations in the cell are used.
    Every sampled cell is returned; 'sufficient' is True where at least
    min_drifters drifters qualify (cells without enough data are kept so that
    the maps can show them as hatched)."""
    kept = per[per["n_obs"] >= min_obs]
    named = {c: (c, "mean") for c in cols}
    g = (kept.groupby(CELL)
             .agg(**named, n_drifters=("drifter_id", "nunique"), n_obs=("n_obs", "sum"))
             .reset_index())
    g = per[CELL].drop_duplicates().merge(g, on=CELL, how="left")
    g[["n_drifters", "n_obs"]] = g[["n_drifters", "n_obs"]].fillna(0).astype(int)
    g["sufficient"] = g["n_drifters"] >= min_drifters
    g["lon"] = g["lon0"] + GRID / 2
    g["lat"] = g["lat0"] + GRID / 2
    return g.sort_values(CELL, ignore_index=True)


def velocity_grid(per, min_obs=MIN_OBS_PER_DRIFTER, min_drifters=MIN_DRIFTERS):
    """Gridded mean current: eastward (ve) and northward (vn) velocity in m/s,
    plus the speed and direction of the mean current."""
    g = cell_grid(per, ["ve", "vn"], min_obs, min_drifters)
    g["speed"] = np.hypot(g["ve"], g["vn"])
    g["direction_deg"] = np.degrees(np.arctan2(g["vn"], g["ve"]))   # 0° = flowing east, 90° = north
    return g


def add_bootstrap_se(grid, per, seed=SEED):
    """Add an uncertainty (standard error) to each mapped cell, and a confidence label.

    The standard error is estimated by bootstrapping: the cell's drifters are
    resampled with replacement N_BOOT times and the mean recomputed each time;
    the spread of those means is the standard error. Resampling drifters rather
    than hourly observations is important, because consecutive hourly readings
    from the same drifter are not independent.

    With n drifters, the bootstrap underestimates the standard error by a factor
    of sqrt((n - 1) / n), which matters for cells with only a few drifters (18%
    too small at n = 3). The result is scaled up by sqrt(n / (n - 1)) to remove
    this bias.

    Confidence: "high" = HIGH_CONF_DRIFTERS or more drifters, "low" = fewer but
    still mapped, "insufficient" = not mapped."""
    rng = np.random.default_rng(seed)
    kept = per[per["n_obs"] >= MIN_OBS_PER_DRIFTER]
    rows = []
    for key, grp in kept.groupby(CELL):
        n = len(grp)
        if n < MIN_DRIFTERS:
            continue
        idx = rng.integers(0, n, size=(N_BOOT, n))
        bve = grp["ve"].to_numpy()[idx].mean(axis=1)
        bvn = grp["vn"].to_numpy()[idx].mean(axis=1)
        scale = np.sqrt(n / (n - 1))       # small-sample correction (see above)
        rows.append((*key, scale * bve.std(ddof=1), scale * bvn.std(ddof=1),
                     scale * np.hypot(bve, bvn).std(ddof=1)))
    se = pd.DataFrame(rows, columns=CELL + ["se_ve", "se_vn", "se_speed"])
    g = grid.merge(se, on=CELL, how="left")
    g["confidence"] = np.select(
        [~g["sufficient"], g["n_drifters"] >= HIGH_CONF_DRIFTERS], ["insufficient", "high"], "low")
    return g


# ============================================================
# Monsoon reversal
# ============================================================
# For each cell mapped in both seasons, compare the direction of the Southwest
# and Northeast mean currents. A change of more than 90° counts as a reversal.

def in_eicc(lon0):
    """True for cells inside the approximate EICC longitude band."""
    return (lon0 >= EICC_LON[0]) & (lon0 < EICC_LON[1])


def angle_between(u1, v1, u2, v2):
    """Smallest angle between two current directions, from 0° (same direction)
    to 180° (exactly opposite)."""
    d = np.degrees(np.arctan2(v2, u2) - np.arctan2(v1, u1))
    return np.abs((d + 180) % 360 - 180)


# Categories of direction change used in the reversal map and tables
ANGLE_BINS = [0, 45, 90, 135, 180]
ANGLE_LABELS = ["little change (<45)", "moderate (45-90)", "strong (90-135)", "near-opposite (>135)"]
ANGLE_COLORS = ["#2166ac", "#92c5de", "#f4a582", "#b2182b"]   # blues = not reversed, reds = reversed


def paired_cells(grid):
    """Cells with enough data in BOTH monsoons, side by side (_sw / _ne columns),
    with the change in current direction from Southwest to Northeast monsoon.
    A cell is 'reversed' if the direction changes by more than 90°."""
    g = grid[grid["sufficient"]].set_index(["lon0", "lat0"])
    cols = ["ve", "vn", "speed", "n_drifters"]
    both = (g[g["monsoon"] == "Southwest"][cols]
            .join(g[g["monsoon"] == "Northeast"][cols], lsuffix="_sw", rsuffix="_ne", how="inner")
            .reset_index())
    both["lon"] = both["lon0"] + GRID / 2
    both["lat"] = both["lat0"] + GRID / 2
    both["angle_change"] = angle_between(both["ve_sw"], both["vn_sw"], both["ve_ne"], both["vn_ne"])
    both["reversed"] = both["angle_change"] > 90
    both["category"] = pd.cut(both["angle_change"], ANGLE_BINS, include_lowest=True, labels=ANGLE_LABELS)
    both["in_eicc_band"] = in_eicc(both["lon0"])
    return both


def bootstrap_reversal(per, eicc_only=False, seed=SEED, thresholds=(90, 135)):
    """95% confidence interval for the percentage of cells that reverse.

    Whole drifters are resampled with replacement N_BOOT times (a drifter drawn
    twice counts twice). Each time, the cell means are rebuilt and the
    percentage of cells whose direction changes by more than each threshold is
    recalculated. With eicc_only=True only the EICC band is used.

    The set of cells is fixed: it is the cells mapped in both seasons in the
    original data (at least MIN_DRIFTERS drifters in each). If cells were
    instead re-selected in every resample, cells near the drifter minimum would
    drop in and out, and the interval would mix real uncertainty with changes
    in which cells are counted. A cell is skipped in a resample only if none of
    its drifters were drawn for one of the seasons.

    Returns one row per threshold: [2.5th percentile, median, 97.5th percentile]."""
    q = per[per["n_obs"] >= MIN_OBS_PER_DRIFTER]
    if eicc_only:
        q = q[in_eicc(q["lon0"])]
    ids, drifter_idx = np.unique(q["drifter_id"].to_numpy(), return_inverse=True)
    cells, cell_idx = np.unique(q[["lon0", "lat0"]].to_numpy(), axis=0, return_inverse=True)
    code = cell_idx.ravel() * 2 + (q["monsoon"].to_numpy() == "Northeast")   # one slot per (cell, season)
    n_slots = len(cells) * 2
    ve, vn = q["ve"].to_numpy(), q["vn"].to_numpy()

    # Cells mapped in both seasons in the original data: the fixed set used throughout
    mapped = (np.bincount(code, minlength=n_slots) >= MIN_DRIFTERS).reshape(-1, 2).all(axis=1)

    rng = np.random.default_rng(seed)
    results = []
    for _ in range(N_BOOT):
        # w = how many times each drifter was drawn in this resample (0, 1, 2, ...);
        # cell means are then weighted averages using these counts
        w = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))[drifter_idx]
        w_sum = np.bincount(code, w, n_slots)
        with np.errstate(invalid="ignore", divide="ignore"):
            mve = (np.bincount(code, w * ve, n_slots) / w_sum).reshape(-1, 2)
            mvn = (np.bincount(code, w * vn, n_slots) / w_sum).reshape(-1, 2)
        both = mapped & (w_sum > 0).reshape(-1, 2).all(axis=1)
        if both.any():
            angle = angle_between(mve[both, 0], mvn[both, 0], mve[both, 1], mvn[both, 1])
            results.append([(angle > t).mean() * 100 for t in thresholds])
    if not results:
        return np.full((len(thresholds), 3), np.nan)
    return np.percentile(results, [2.5, 50, 97.5], axis=0).T


# ============================================================
# Analysis steps
# ============================================================
# Each function below produces one part of the report, with its figures and
# tables. run() calls them in order.

def explore(ds, obs):
    """Overview of the cleaned data (all months, all drifters): when and where
    drifters sampled, the share that were drogued, and current speeds by drogue
    status. This describes the full data, so it is the same for every dataset."""
    section("COVERAGE")
    coverage = (obs.groupby("monsoon")
                   .agg(observations=("time", "size"), unique_drifters=("drifter_id", "nunique"),
                        drogued_pct=("drogue", "mean"))
                   .reindex(SEASONS + ["Inter-monsoon"]))
    coverage["drogued_pct"] *= 100
    print(coverage.round(1))
    save_table(coverage, "coverage_by_monsoon.csv", index=True)

    # Observations per month and per year
    monthly = obs["month"].value_counts().sort_index()
    yearly = obs["year"].value_counts().sort_index()
    save_table(pd.DataFrame({"observations": monthly}), "observations_by_month.csv", index=True)
    save_table(pd.DataFrame({"observations": yearly}), "observations_by_year.csv", index=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), constrained_layout=True)
    colors = ["coral" if m in (6, 7, 8, 9) else "steelblue" if m in (12, 1, 2) else "lightgrey"
              for m in monthly.index]
    axes[0].bar(monthly.index, monthly.values, color=colors)
    axes[0].set_xticks(range(1, 13))
    axes[0].set(xlabel="Month", ylabel="Observations", title="Observations per month")
    axes[0].legend(handles=[Patch(color="coral", label="Southwest"), Patch(color="steelblue", label="Northeast"),
                            Patch(color="lightgrey", label="Inter-monsoon")])
    axes[1].bar(yearly.index, yearly.values, color="steelblue")
    axes[1].set(xlabel="Year", ylabel="Observations", title="Observations per year")
    save_fig(fig, "01_temporal_coverage.png")

    # Drogue status by monsoon
    drogue_pct = (pd.crosstab(obs["monsoon"], obs["drogue"], normalize="index") * 100)
    drogue_pct = drogue_pct.reindex(SEASONS + ["Inter-monsoon"]).rename(columns={True: "Drogued", False: "Undrogued"})
    ax = drogue_pct[["Drogued", "Undrogued"]].plot(kind="bar", stacked=True, figsize=(7, 4),
                                                   color=["steelblue", "lightgrey"])
    ax.set(ylabel="% of observations", xlabel="", title="Drogue status by monsoon period")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    save_fig(ax.figure, "02_drogue_by_monsoon.png")

    # Speed by drogue status. Undrogued drifters moving faster than drogued ones
    # supports the view that their velocities are affected by wind, and is the
    # reason drogued drifters are the default dataset.
    section("CURRENT SPEED BY DROGUE STATUS")
    season_obs = obs[obs["monsoon"].isin(SEASONS)].assign(speed=lambda d: np.hypot(d["ve"], d["vn"]))
    speed_summary = (season_obs.groupby(["monsoon", "drogue"])["speed"]
                     .describe(percentiles=[0.25, 0.5, 0.75]).round(3))
    print(speed_summary)
    save_table(speed_summary, "speed_by_drogue_status.csv", index=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, constrained_layout=True)
    for ax, season in zip(axes, SEASONS):
        s = season_obs[season_obs["monsoon"] == season]
        ax.boxplot([s.loc[s["drogue"], "speed"], s.loc[~s["drogue"], "speed"]], showfliers=False)
        ax.set_xticks([1, 2], labels=["Drogued", "Undrogued"])
        ax.set_title(f"{season} monsoon")
    axes[0].set_ylabel("Current speed (m/s)")
    fig.suptitle("Current speed by drogue status")
    save_fig(fig, "03_speed_by_drogue.png")

    # Sampling maps on a fine 50 x 50 grid: number of hourly observations vs
    # number of different drifters per cell. Observation counts can be inflated
    # by a single drifter circling in one place; drifter counts show how many
    # independent samples each area actually has.
    bins = [np.linspace(*LON_RANGE, 51), np.linspace(*LAT_RANGE, 51)]
    counts, xe, ye = np.histogram2d(obs["lon"], obs["lat"], bins=bins)
    ix = np.clip(np.searchsorted(xe, obs["lon"], side="right") - 1, 0, 49)
    iy = np.clip(np.searchsorted(ye, obs["lat"], side="right") - 1, 0, 49)
    # Count each drifter once per cell: keep only its first observation there
    first_visit = ~pd.DataFrame({"d": obs["drifter_id"].to_numpy(), "x": ix, "y": iy}).duplicated().to_numpy()
    unique, _, _ = np.histogram2d(obs["lon"][first_visit], obs["lat"][first_visit], bins=bins)

    panels = [
        (counts, "YlOrBr", "Observations per cell", "Observation density"),
        (unique, "YlGnBu", "Distinct drifters per cell", "Distinct drifter coverage"),
    ]

    def draw_density(fig, ax, grid, cmap_name, label, title, legend=True):
        g = np.where(grid == 0, np.nan, grid)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("white")
        im = ax.pcolormesh(xe, ye, g.T, cmap=cmap, shading="auto")
        add_land(ax)
        fig.colorbar(im, ax=ax, label=label)
        setup_map(ax, title)
        if legend:
            add_land_legend(ax)

    # Each panel on its own, then side by side
    for panel, fname in zip(panels, ["04a_observation_density.png", "04b_distinct_drifter_density.png"]):
        fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
        draw_density(fig, ax, *panel)
        save_fig(fig, fname)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    for ax, panel in zip(axes, panels):
        draw_density(fig, ax, *panel, legend=ax is axes[0])
    save_fig(fig, "04_density_and_unique_drifters.png")

    # Drifter tracks, with each drifter's first position in the region marked in red
    fig, ax = plt.subplots(figsize=(10, 8))
    for _, d in obs.groupby("drifter_id", sort=False):
        ax.plot(d["lon"], d["lat"], linewidth=0.4, alpha=0.15, color="blue")
        ax.scatter(d["lon"].iloc[0], d["lat"].iloc[0], s=12, color="red", alpha=0.8)
    add_land(ax)
    add_land_legend(ax)
    setup_map(ax, "Drifter trajectories and first fix in region")
    save_fig(fig, "05_trajectories.png")


def threshold_sensitivity(per):
    """Repeat the mapping and reversal calculation for a range of minimum-drifter
    and minimum-hours rules, to show how the results depend on these choices.
    This is how the chosen thresholds (MIN_DRIFTERS, MIN_OBS_PER_DRIFTER) were selected."""
    section(f"THRESHOLD SENSITIVITY ({dataset_label()})")
    rows = []
    for md in SENS_MIN_DRIFTERS:
        for mo in SENS_MIN_OBS:
            g = velocity_grid(per, min_obs=mo, min_drifters=md)
            p = paired_cells(g)
            kept = g[g["sufficient"]]["monsoon"].value_counts()
            rows.append({
                "min_drifters": md, "min_obs_per_drifter": mo,
                "sw_cells": kept.get("Southwest", 0), "ne_cells": kept.get("Northeast", 0),
                "cells_both": len(p), "reversed_pct_bay": p["reversed"].mean() * 100,
                "cells_both_eicc": int(p["in_eicc_band"].sum()),
                "reversed_pct_eicc": p.loc[p["in_eicc_band"], "reversed"].mean() * 100,
            })
    table = pd.DataFrame(rows)
    print(table.round(1).to_string(index=False))
    save_table(table, "threshold_sensitivity.csv")


def circulation_figures(grid):
    """Maps of mean surface current for each monsoon (main result) and the
    uncertainty map, for the active dataset."""
    section("CIRCULATION MAPS")
    vmax = np.ceil(grid.loc[grid["sufficient"], "speed"].max() * 10) / 10
    norm, cmap = Normalize(0, vmax), plt.get_cmap("YlGnBu")

    # Main result: mean current, one map per season
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
    for ax, season in zip(axes, SEASONS):
        season_cell_map(ax, grid[grid["monsoon"] == season], "speed", cmap, norm,
                        f"{season} monsoon", arrows=True)
    add_land_legend(axes[0], [INSUFFICIENT_PATCH])
    add_colorbar(fig, axes, cmap, norm, "Mean current speed (m/s)")
    fig.suptitle(f"Drifter-weighted mean surface circulation ({dataset_label()})", fontsize=15)
    save_fig(fig, "06_circulation.png", dpi=300)

    # Uncertainty: bootstrap standard error of the speed.
    # Colour scale capped at the 95th percentile so a few extreme cells do not
    # wash out the rest; dots mark high-confidence cells.
    se_max = grid["se_speed"].quantile(0.95)
    se_norm, se_cmap = Normalize(0, se_max), plt.get_cmap("Oranges")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
    for ax, season in zip(axes, SEASONS):
        season_cell_map(ax, grid[grid["monsoon"] == season], "se_speed", se_cmap, se_norm,
                        f"{season} monsoon")
        g = grid[(grid["monsoon"] == season) & (grid["confidence"] == "high")]
        ax.scatter(g["lon"], g["lat"], marker=".", color="black", s=20, zorder=3)
    add_land_legend(axes[0], [INSUFFICIENT_PATCH, plt.Line2D([], [], marker=".", color="black",
                              linestyle="", label=f"High confidence (>={HIGH_CONF_DRIFTERS} drifters)")])
    add_colorbar(fig, axes, se_cmap, se_norm, "Bootstrap standard error of speed (m/s)")
    fig.suptitle(f"Uncertainty of the mean current ({dataset_label()})", fontsize=15)
    save_fig(fig, "08_uncertainty.png")


def reversal_analysis(per, grid):
    """Headline result: the percentage of cells whose current reverses (> 90°)
    and strongly reverses (> 135°) between monsoons, with 95% confidence
    intervals, for the whole Bay and the EICC band. Also draws the reversal map
    and writes its caption."""
    section(f"MONSOON REVERSAL (SW vs NE direction differs by > 90 deg, {N_BOOT} drifter bootstrap)")
    rows = []
    cells = paired_cells(grid)
    for region, eicc_only in [("Whole Bay of Bengal", False),
                              (f"Approx. EICC band {EICC_LON[0]:.0f}-{EICC_LON[1]:.0f}E", True)]:
        r = cells[cells["in_eicc_band"]] if eicc_only else cells
        (lo, med, hi), (lo135, med135, hi135) = bootstrap_reversal(per, eicc_only=eicc_only)
        rows.append({"dataset": ACTIVE, "region": region, "cells_both_seasons": len(r),
                     "cells_reversed": int(r["reversed"].sum()),
                     "reversed_pct": r["reversed"].mean() * 100,
                     "boot_median_pct": med, "ci95_low_pct": lo, "ci95_high_pct": hi,
                     "cells_gt135": int((r["angle_change"] > 135).sum()),
                     "gt135_pct": (r["angle_change"] > 135).mean() * 100,
                     "gt135_boot_median_pct": med135, "gt135_ci95_low_pct": lo135,
                     "gt135_ci95_high_pct": hi135})
    summary = pd.DataFrame(rows)
    print(summary.round(1).to_string(index=False))
    save_table(summary, "reversal_summary.csv")

    print("\nCells by direction-change category:")
    print(pd.crosstab(cells["category"], cells["in_eicc_band"].map({True: "EICC band", False: "rest of bay"}),
                      margins=True))
    save_table(cells, f"reversal_cells_{ACTIVE}.csv")

    # Southern edge of the region (5-6°N): how well sampled are these cells?
    south = cells[cells["lat0"] == LAT_RANGE[0]][["lon0", "lat0", "angle_change", "category",
                                                  "n_drifters_sw", "n_drifters_ne"]]
    print(f"\nSouthern-edge cells ({LAT_RANGE[0]:.0f}-{LAT_RANGE[0] + GRID:.0f}N):")
    print(south.round(0).to_string(index=False))
    save_table(south, "reversal_southern_edge.csv")

    # Reversal map: direction change per cell, in four categories so that the
    # 90° reversal threshold is the split between blue (not reversed) and red
    # (reversed). The headline numbers are written to reversal_caption.txt
    # rather than the figure title. Ocean cells without data in both seasons are
    # hatched; land is painted white underneath so the hatching stops at the coast.
    norm, cmap = BoundaryNorm(ANGLE_BINS, len(ANGLE_COLORS)), ListedColormap(ANGLE_COLORS)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    no_data_style = dict(facecolor="lightgrey", edgecolor="grey", hatch="//", alpha=0.6, linewidth=0)
    ax.add_patch(plt.Rectangle((LON_RANGE[0], LAT_RANGE[0]), np.ptp(LON_RANGE), np.ptp(LAT_RANGE),
                               **no_data_style, zorder=-1))
    if _get_land():
        ax.add_collection(PolyCollection(_get_land(), facecolors="white", edgecolors="none",
                                         zorder=-0.5), autolim=False)
    draw_cells(ax, cells, cells["angle_change"], cmap, norm)
    add_land(ax)
    setup_map(ax, f"SW-to-NE change in current direction ({dataset_label()})")
    add_land_legend(ax, [Patch(**no_data_style, label="No reliable data in one or both seasons")])
    cbar = add_colorbar(fig, ax, cmap, norm, "Direction change SW to NE (>90° = reversed)")
    cbar.set_ticks([(lo + hi) / 2 for lo, hi in zip(ANGLE_BINS[:-1], ANGLE_BINS[1:])],
                   labels=[f"{lo}-{hi}°\n{name.split(' (')[0]}"
                           for lo, hi, name in zip(ANGLE_BINS[:-1], ANGLE_BINS[1:], ANGLE_LABELS)])
    cbar.ax.tick_params(length=0)
    cbar.ax.axhline(90, color="black", linewidth=2)
    save_fig(fig, "09_reversal_map.png", dpi=300)

    caption = reversal_caption(summary)
    print("\nCaption for the reversal map:\n" + caption)
    with open(os.path.join(OUT_DIR, "reversal_caption.txt"), "w", encoding="utf-8") as f:
        f.write(caption + "\n")
    return summary


def reversal_caption(rows):
    """Plain-language caption for the reversal map.

    For reference: if current directions in the two seasons were unrelated, about
    50% of cells would change by more than 90° and 25% by more than 135° purely
    by chance. Where the confidence interval reaches down to these levels, the
    caption says the result is not clearly different from chance."""
    lines = []
    for _, r in rows.iterrows():
        def part(pct, med, lo, hi, chance):
            s = f"{pct:.0f}% (bootstrap median {med:.0f}%, 95% CI {lo:.0f}-{hi:.0f}%)"
            return s + (f" - not clearly different from the {chance}% expected by chance" if lo <= chance else "")
        lines.append(f"{r['region']} ({r['cells_both_seasons']} cells): "
                     f"> 90° change in {part(r.reversed_pct, r.boot_median_pct, r.ci95_low_pct, r.ci95_high_pct, 50)}; "
                     f"> 135° change in {part(r.gt135_pct, r.gt135_boot_median_pct, r.gt135_ci95_low_pct, r.gt135_ci95_high_pct, 25)}.")
    return "\n".join(lines)


# ============================================================
# Interannual baseline: how much change happens without a monsoon contrast?
# ============================================================
# Some cells would show a large direction change (here > 135°, near-opposite flow,
# with > 90° also reported) between any two groups of years, because
# eddies move, years differ and cells with few drifters are noisy. To measure
# that background, the years are split at random into two halves, A and B, and
# four half-data maps are built: SW-A, SW-B, NE-A and NE-B.
#   - Same season, different years (SW-A vs SW-B, NE-A vs NE-B): change with
#     no monsoon contrast (interannual variability + sampling noise).
#   - Different season, different years (SW-A vs NE-B, SW-B vs NE-A): the same
#     comparison with the monsoon contrast added.
# Both use the same cells and the same amount of data, so the difference
# between them is the part of the reversal that the monsoon is responsible for.
# The split is repeated N_BOOT times. Splitting drifters instead of years gives
# the background from sampling noise alone; the gap between the two background
# levels is the contribution of genuine year-to-year variability.

def season_years(obs):
    """The monsoon year of each drifter's record in each season. December is
    counted with the following January-February, so that one Northeast monsoon
    is one season-year."""
    year = obs["year"] + ((obs["monsoon"] == "Northeast") & (obs["month"] == 12))
    return (obs.assign(season_year=year)
               .groupby(["monsoon", "drifter_id"])["season_year"].min()
               .reset_index())


def split_half_reversal(per, years, split="year", eicc_only=False, min_drifters=BASELINE_MIN_DRIFTERS,
                        threshold=135, seed=SEED):
    """% of cells changing direction by more than `threshold` degrees between
    half-data maps, for many random splits of the years (split="year") or of
    the drifters (split="drifter"). A cell is used only if all four half-data
    maps have at least min_drifters drifters there.

    Returns one row per split: [cells used, SW vs NE, SW vs SW, NE vs NE]."""
    q = per[per["n_obs"] >= MIN_OBS_PER_DRIFTER].merge(years, on=["monsoon", "drifter_id"])
    if eicc_only:
        q = q[in_eicc(q["lon0"])]
    cells, cell_idx = np.unique(q[["lon0", "lat0"]].to_numpy(), axis=0, return_inverse=True)
    season = (q["monsoon"].to_numpy() == "Northeast").astype(int)
    units, unit_idx = np.unique(q["season_year" if split == "year" else "drifter_id"].to_numpy(),
                                return_inverse=True)
    ve, vn = q["ve"].to_numpy(), q["vn"].to_numpy()
    n_slots = len(cells) * 4          # one slot per cell x {SW-A, SW-B, NE-A, NE-B}

    rng = np.random.default_rng(seed)
    results = []
    for _ in range(N_BOOT):
        in_b = rng.permutation(len(units)) < len(units) // 2
        code = cell_idx.ravel() * 4 + season * 2 + in_b[unit_idx.ravel()]
        n = np.bincount(code, minlength=n_slots).reshape(-1, 4)
        with np.errstate(invalid="ignore", divide="ignore"):
            mve = (np.bincount(code, ve, n_slots) / n.ravel()).reshape(-1, 4)
            mvn = (np.bincount(code, vn, n_slots) / n.ravel()).reshape(-1, 4)
        ok = (n >= min_drifters).all(axis=1)
        if not ok.any():
            continue

        def pct(i, j):
            return (angle_between(mve[ok, i], mvn[ok, i], mve[ok, j], mvn[ok, j]) > threshold).mean() * 100

        # slot order: 0 = SW-A, 1 = SW-B, 2 = NE-A, 3 = NE-B
        results.append([ok.sum(), (pct(0, 3) + pct(1, 2)) / 2, pct(0, 1), pct(2, 3)])
    return np.array(results)


def interannual_baseline(data, per):
    """Compare the SW-vs-NE reversal with the change seen between different
    years of the same season, for the whole Bay and the EICC band. `data` is
    the active dataset's observations and `per` its per-drifter cell means."""
    section(f"INTERANNUAL BASELINE (same season, different years; {N_BOOT} random splits)")
    years = season_years(data)
    print("Season-years with data: "
          + ", ".join(f"{s} {years.loc[years['monsoon'] == s, 'season_year'].nunique()}" for s in SEASONS))

    regions = [("Whole Bay of Bengal", False),
               (f"Approx. EICC band {EICC_LON[0]:.0f}-{EICC_LON[1]:.0f}E", True)]
    rows, draws = [], {}
    for threshold in BASELINE_THRESHOLDS:
        for split, split_label in [("year", "years"), ("drifter", "drifters")]:
            for region, eicc_only in regions:
                r = split_half_reversal(per, years, split=split, eicc_only=eicc_only, threshold=threshold)
                if len(r) == 0:
                    continue
                draws[(threshold, split, region)] = r
                excess = r[:, 1] - (r[:, 2] + r[:, 3]) / 2
                q = lambda x: np.percentile(x, [50, 2.5, 97.5])
                rows.append({
                    "threshold_deg": threshold, "split_by": split_label, "region": region,
                    "cells_median": int(np.median(r[:, 0])),
                    **{f"{name}_{stat}": v
                       for name, x in [("sw_vs_ne_pct", r[:, 1]), ("sw_vs_sw_pct", r[:, 2]),
                                       ("ne_vs_ne_pct", r[:, 3]), ("monsoon_excess_pct", excess)]
                       for stat, v in zip(["median", "ci95_low", "ci95_high"], q(x))},
                    "monsoon_share_pct": 100 * np.median(excess) / np.median(r[:, 1]),
                })
    table = pd.DataFrame(rows)
    print(table.round(1).to_string(index=False))
    save_table(table, "interannual_baseline.csv")

    for _, t in table[table["split_by"] == "years"].iterrows():
        print(f"\n> {t['threshold_deg']}° | {t['region']} (median {t['cells_median']} cells, half the years "
              f"in each map, >= {BASELINE_MIN_DRIFTERS} drifter(s) per cell per map):\n"
              f"  SW vs NE: {t['sw_vs_ne_pct_median']:.0f}% of cells change by > {t['threshold_deg']}°\n"
              f"  Same season, different years: SW {t['sw_vs_sw_pct_median']:.0f}%, "
              f"NE {t['ne_vs_ne_pct_median']:.0f}%\n"
              f"  Change attributable to the monsoon: {t['monsoon_excess_pct_median']:.0f} percentage points "
              f"(95% range {t['monsoon_excess_pct_ci95_low']:.0f} to {t['monsoon_excess_pct_ci95_high']:.0f}), "
              f"i.e. about {t['monsoon_share_pct']:.0f}% of the SW-vs-NE change")

    # Figure: distribution over random year splits of each comparison, one row per threshold.
    # The dotted line is the level expected if directions were completely random.
    fig, axes = plt.subplots(len(BASELINE_THRESHOLDS), len(regions),
                             figsize=(6.5 * len(regions), 5 * len(BASELINE_THRESHOLDS)),
                             sharey=True, constrained_layout=True, squeeze=False)
    for row, threshold in enumerate(BASELINE_THRESHOLDS):
        for ax, (region, _) in zip(axes[row], regions):
            r = draws.get((threshold, "year", region))
            if r is None:
                ax.set_axis_off()
                continue
            box = ax.boxplot([r[:, 1], r[:, 2], r[:, 3]], showfliers=False, patch_artist=True, widths=0.6)
            for patch, colour in zip(box["boxes"], ["firebrick", "coral", "steelblue"]):
                patch.set(facecolor=colour, alpha=0.6)
            ax.set_xticks([1, 2, 3], labels=["SW vs NE\n(monsoon contrast)", "SW vs SW\n(different years)",
                                             "NE vs NE\n(different years)"])
            chance = 100 * (180 - threshold) / 180
            ax.axhline(chance, color="grey", linestyle=":", label=f"Random directions ({chance:.0f}%)")
            ax.set(title=f"Change > {threshold}°: {region}\n(median {int(np.median(r[:, 0]))} cells)",
                   ylim=(0, 100))
            ax.legend(loc="upper right", fontsize=9)
        axes[row, 0].set_ylabel(f"% of cells changing direction by > {threshold}°")
    fig.suptitle("Monsoon reversal compared with change between different years of the same season", fontsize=14)
    save_fig(fig, "18_interannual_baseline.png")

    # Sensitivity: larger cells and stricter drifter minimums (year split only)
    print("\nSensitivity to cell size and drifter minimum (random year splits, medians):")
    sens = []
    for cell in BASELINE_SENS_CELLS:
        coarse = data.assign(
            lon0=LON_RANGE[0] + cell * np.floor((data["lon0"] - LON_RANGE[0]) / cell),
            lat0=LAT_RANGE[0] + cell * np.floor((data["lat0"] - LAT_RANGE[0]) / cell))
        per_coarse = per_drifter_means(coarse)
        for threshold in BASELINE_THRESHOLDS:
            for min_d in BASELINE_SENS_MIN:
                for region, eicc_only in [("Whole Bay", False), ("EICC band", True)]:
                    r = split_half_reversal(per_coarse, years, eicc_only=eicc_only, min_drifters=min_d,
                                            threshold=threshold)
                    if len(r) == 0:
                        continue
                    excess = r[:, 1] - (r[:, 2] + r[:, 3]) / 2
                    sens.append({"threshold_deg": threshold, "cell_deg": cell, "min_drifters": min_d,
                                 "region": region, "cells_median": int(np.median(r[:, 0])),
                                 "sw_vs_ne_pct": np.median(r[:, 1]), "sw_vs_sw_pct": np.median(r[:, 2]),
                                 "ne_vs_ne_pct": np.median(r[:, 3]), "monsoon_excess_pct": np.median(excess),
                                 "excess_ci95_low": np.percentile(excess, 2.5),
                                 "excess_ci95_high": np.percentile(excess, 97.5)})
    sens = pd.DataFrame(sens)
    print(sens.round(1).to_string(index=False))
    save_table(sens, "interannual_baseline_sensitivity.csv")
    return table


def compare_pair(grid_a, grid_b, a, b):
    """Cell-by-cell comparison of two datasets' mean currents, using the cells
    mapped in both. Dataset `a` is the reference (e.g. drogued) and `b` the one
    compared with it. Reports how much speed and direction differ, and whether
    the differences shrink where the reference has more drifters (if so, they
    are mostly small-sample noise rather than a real difference)."""
    name_a, name_b = DATASETS[a], DATASETS[b]
    section(f"CELL-BY-CELL: {name_b} vs {name_a}, cells mapped in both")
    cols = CELL + ["ve", "vn", "speed", "n_drifters"]
    m = (grid_a[grid_a["sufficient"]][cols + ["lon", "lat"]]
         .merge(grid_b[grid_b["sufficient"]][cols], on=CELL, suffixes=("_a", "_b")))
    m.insert(0, "dataset_a", a)
    m.insert(1, "dataset_b", b)
    m["speed_diff"] = m["speed_b"] - m["speed_a"]
    m["abs_speed_diff"] = m["speed_diff"].abs()
    m["pct_speed_diff"] = np.where(m["speed_a"] >= MIN_SPEED_FOR_PCT,
                                   m["abs_speed_diff"] / m["speed_a"] * 100, np.nan)
    m["angle_diff"] = angle_between(m["ve_a"], m["vn_a"], m["ve_b"], m["vn_b"])
    m["sample_group"] = pd.cut(m["n_drifters_a"], [0, 4, 9, np.inf], labels=["3-4", "5-9", "10+"])
    if m.empty:
        print("No cells are mapped in both datasets.")
        return m

    print(m.groupby("monsoon")[["abs_speed_diff", "pct_speed_diff", "angle_diff"]].describe().T.round(3))
    for season in SEASONS:
        d = m[m["monsoon"] == season]
        shares = ", ".join(f">{t} deg: {(d['angle_diff'] > t).mean() * 100:.1f}%" for t in (15, 30, 45, 90))
        print(f"\n{season}: direction change {shares}")
        print(f"  corr(drifters in {a}, |speed diff|) = {d['n_drifters_a'].corr(d['abs_speed_diff']):.3f}")
        print(f"  corr(drifters in {a}, direction diff) = {d['n_drifters_a'].corr(d['angle_diff']):.3f}")
    print(f"\nBy number of {a} drifters in the cell:")
    print(m.groupby(["monsoon", "sample_group"], observed=True)[["abs_speed_diff", "angle_diff"]]
           .agg(["count", "median", "mean", "max"]).round(3))
    save_table(m, f"cells_{b}_vs_{a}.csv")

    # Differences against the number of reference drifters in the cell
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, constrained_layout=True)
    for row, season in enumerate(SEASONS):
        d = m[m["monsoon"] == season]
        axes[row, 0].scatter(d["n_drifters_a"], d["abs_speed_diff"], alpha=0.5)
        axes[row, 0].set(ylabel="|Speed difference| (m/s)", title=f"{season}: speed")
        axes[row, 1].scatter(d["n_drifters_a"], d["angle_diff"], alpha=0.5, color="darkorange")
        axes[row, 1].set(ylabel="Direction difference (deg)", title=f"{season}: direction")
    for ax in axes[1]:
        ax.set_xlabel(f"Drifters in cell ({name_a})")
    fig.suptitle(f"{name_b.capitalize()} vs {name_a}, against sample size", fontsize=15)
    save_fig(fig, f"10_scatter_{b}_vs_{a}.png")

    # Maps of where the differences are (green = small, red = large)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    panels = [("pct_speed_diff", Normalize(0, 50), "% speed difference"),
              ("angle_diff", Normalize(0, 90), "Direction difference (deg)")]
    for row, season in enumerate(SEASONS):
        d = m[m["monsoon"] == season].assign(sufficient=True)
        for col, (value, norm, title) in enumerate(panels):
            season_cell_map(axes[row, col], d, value, plt.get_cmap("RdYlGn_r"), norm, f"{season}: {title}")
    for col, (_, norm, title) in enumerate(panels):
        add_colorbar(fig, axes[:, col], plt.get_cmap("RdYlGn_r"), norm, title)
    add_land_legend(axes[0, 0])
    fig.suptitle(f"Where the mean current differs: {name_b} vs {name_a}", fontsize=15)
    save_fig(fig, f"11_map_{b}_vs_{a}.png")
    return m


def sst_analysis(data):
    """Gridded mean sea surface temperature for each monsoon, built the same way
    as the currents (drifter-weighted, same mapping rules), after removing SST
    readings with a reported error above ERR_SST_MAX."""
    section(f"SEA SURFACE TEMPERATURE ({dataset_label()})")
    err = data["err_sst"].dropna()
    print(err.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(3))
    print(f"err_sst > {ERR_SST_MAX}: {(err > ERR_SST_MAX).mean() * 100:.2f}% of non-null observations dropped")

    sst_obs = data.dropna(subset=["sst", "err_sst"])
    sst_obs = sst_obs[sst_obs["err_sst"] <= ERR_SST_MAX]
    grid = cell_grid(per_drifter_means(sst_obs, ["sst"]), ["sst"])
    ok = grid[grid["sufficient"]]
    print(ok.groupby("monsoon")["sst"].describe().round(2))
    save_table(grid, f"sst_grid_{ACTIVE}.csv")

    norm, cmap = Normalize(ok["sst"].min(), ok["sst"].max()), plt.get_cmap("YlOrRd")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
    for ax, season in zip(axes, SEASONS):
        season_cell_map(ax, grid[grid["monsoon"] == season], "sst", cmap, norm, f"{season} monsoon")
    add_land_legend(axes[0], [INSUFFICIENT_PATCH])
    add_colorbar(fig, axes, cmap, norm, "Mean sea surface temperature (°C)")
    fig.suptitle(f"Drifter-weighted mean SST ({dataset_label()}, err_sst <= {ERR_SST_MAX} °C)", fontsize=15)
    save_fig(fig, "12_sst.png", dpi=300)
    return grid


# ============================================================
# SPATIAL GAUSSIAN KERNEL SMOOTHING — DATA3001
# ============================================================
# Supplementary analysis of the active dataset's surface currents.
# Original gridded estimates and reversal statistics are retained.

KERNEL_BANDWIDTHS_KM = (
    25.0, 40.0, 50.0, 60.0, 75.0, 90.0,
    100.0, 125.0, 175.0, 250.0, 350.0
)
KERNEL_TEST_FRACTION = 0.20
KERNEL_RADIUS_MULTIPLIER = 3.0


def train_test_drifters(per):
    """Split the drifters at random into training (80%) and test (20%) sets.
    Whole drifters are kept together, so no test drifter's data is ever used
    to build the map it is scored against. The same split (fixed by SEED) is
    used by the kernel smoothing and the Gaussian process, so their test
    scores are directly comparable."""
    drifter_ids = np.sort(per["drifter_id"].unique())
    shuffled = np.random.default_rng(SEED).permutation(drifter_ids)
    n_test = max(1, round(len(shuffled) * KERNEL_TEST_FRACTION))
    return shuffled[n_test:], shuffled[:n_test]


def kernel_distance_km(lon_a, lat_a, lon_b, lat_b):
    """Pairwise great-circle distances in kilometres."""
    a_lon = np.radians(np.asarray(lon_a, dtype=float))[:, None]
    a_lat = np.radians(np.asarray(lat_a, dtype=float))[:, None]
    b_lon = np.radians(np.asarray(lon_b, dtype=float))[None, :]
    b_lat = np.radians(np.asarray(lat_b, dtype=float))[None, :]

    hav = np.sin((a_lat - b_lat) / 2) ** 2 + (
        np.cos(a_lat) * np.cos(b_lat)
        * np.sin((a_lon - b_lon) / 2) ** 2
    )
    return 6371.0 * 2 * np.arcsin(np.sqrt(np.clip(hav, 0.0, 1.0)))


def kernel_predict(targets, reference, h_km):
    """Gaussian smoothing of drifter-weighted cell velocity components.

    Only qualifying reference cells are supplied. The result has
    one pair of smoothed eastward/northward velocities per target.
    """
    pred = np.full((len(targets), 2), np.nan, dtype=float)

    if len(targets) == 0 or len(reference) == 0:
        return pred
    if h_km <= 0:
        raise ValueError("Bandwidth must be positive")

    distance = kernel_distance_km(
        targets["lon"], targets["lat"],
        reference["lon"], reference["lat"]
    )

    weights = np.exp(-0.5 * (distance / h_km) ** 2)
    weights[distance > KERNEL_RADIUS_MULTIPLIER * h_km] = 0.0

    denominator = weights.sum(axis=1)
    valid = denominator > 0

    if valid.any():
        values = reference[["ve", "vn"]].to_numpy(dtype=float)
        pred[valid] = (
            weights[valid] @ values
        ) / denominator[valid, None]

    return pred


def kernel_evaluate(heldout_per, reference_per, bandwidths,
                    leave_one_out=False):
    """Evaluate original and smoothed estimates on identical held-out cells.

    When leave_one_out=True, all records from the evaluated drifter
    are removed before constructing the reference grid.
    """
    candidates = (0.0,) + tuple(bandwidths)
    records = []

    groups = heldout_per[
        heldout_per["n_obs"] >= MIN_OBS_PER_DRIFTER
    ].groupby(["monsoon", "drifter_id"], sort=True)

    cached = {}

    for (season, drifter_id), actual in groups:

        if leave_one_out:
            training = reference_per[
                (reference_per["monsoon"] == season)
                & (reference_per["drifter_id"] != drifter_id)
            ]
            reference_grid = velocity_grid(training)

        else:
            if season not in cached:
                training = reference_per[
                    reference_per["monsoon"] == season
                ]
                cached[season] = velocity_grid(training)

            reference_grid = cached[season]

        eligible = reference_grid[
            reference_grid["sufficient"]
        ][["lon0", "lat0", "lon", "lat", "ve", "vn"]]

        if eligible.empty:
            continue

        paired = actual.merge(
            eligible[["lon0", "lat0", "ve", "vn"]],
            on=["lon0", "lat0"],
            how="inner",
            suffixes=("_true", "_raw")
        )

        if paired.empty:
            continue

        truth = paired[
            ["ve_true", "vn_true"]
        ].to_numpy(dtype=float)

        baseline = paired[
            ["ve_raw", "vn_raw"]
        ].to_numpy(dtype=float)

        targets = pd.DataFrame({
            "lon": paired["lon0"] + GRID / 2,
            "lat": paired["lat0"] + GRID / 2
        })

        for h in candidates:
            estimate = (
                baseline if h == 0
                else kernel_predict(targets, eligible, h)
            )

            if not np.isfinite(estimate).all():
                raise AssertionError(
                    f"Missing prediction at valid cell, bandwidth={h}"
                )

            squared_error = ((truth - estimate) ** 2).sum(axis=1)

            records.append({
                "monsoon": season,
                "drifter_id": drifter_id,
                "bandwidth_km": h,
                "n_cells": len(paired),
                "mean_sq_vector_error": float(squared_error.mean())
            })

    return pd.DataFrame(
        records,
        columns=[
            "monsoon", "drifter_id", "bandwidth_km",
            "n_cells", "mean_sq_vector_error"
        ]
    )


def kernel_score_table(prediction_errors):
    """Compute drifter-grouped vector RMSE in m/s."""
    rows = []

    for scope in SEASONS + ["Overall"]:
        if scope == "Overall":
            part = prediction_errors
        else:
            part = prediction_errors[
                prediction_errors["monsoon"] == scope
            ]

        for h, records in part.groupby("bandwidth_km"):
            rmse = np.sqrt(
                records["mean_sq_vector_error"].mean()
            )

            rows.append({
                "scope": scope,
                "bandwidth_km": float(h),
                "rmse_vector_mps": float(rmse),
                "n_drifters": int(
                    records["drifter_id"].nunique()
                ),
                "n_drifter_seasons": len(records),
                "n_heldout_cell_means": int(
                    records["n_cells"].sum()
                )
            })

    return pd.DataFrame(rows)


def kernel_smoothing_analysis(per, grid):
    """Run spatial kernel smoothing, validation and visualisation for the
    active dataset (`per` = per-drifter cell means, `grid` = gridded climatology)."""
    section(f"SPATIAL KERNEL SMOOTHING ({dataset_label()})")

    drifter_ids = np.sort(per["drifter_id"].unique())

    if len(drifter_ids) < 10:
        print("Too few drifters for kernel validation.")
        return None

    train_ids, test_ids = train_test_drifters(per)

    train = per[
        per["drifter_id"].isin(train_ids)
    ].copy()

    test = per[
        per["drifter_id"].isin(test_ids)
    ].copy()

    assert set(train["drifter_id"]).isdisjoint(
        set(test["drifter_id"])
    )

    print(
        f"Drifters: {len(drifter_ids)} total, "
        f"{len(train_ids)} training, {len(test_ids)} testing"
    )

    # --------------------------------------------------------
    # 1. SELECT BANDWIDTH USING TRAINING DRIFTERS
    # --------------------------------------------------------
    print("\nRunning leave-one-drifter-out cross-validation...")

    cv_errors = kernel_evaluate(
        train, train, KERNEL_BANDWIDTHS_KM,
        leave_one_out=True
    )

    if cv_errors.empty:
        print("Insufficient coverage for cross-validation.")
        return

    cv_scores = kernel_score_table(cv_errors)
    save_table(cv_scores, "kernel_bandwidth_cv.csv")

    overall = cv_scores[
        (cv_scores["scope"] == "Overall")
        & (cv_scores["bandwidth_km"] > 0)
    ].sort_values(["rmse_vector_mps", "bandwidth_km"])

    h_best = float(overall.iloc[0]["bandwidth_km"])

    print("\nTRAINING CROSS-VALIDATION:")
    print(cv_scores.round(4).to_string(index=False))
    print(f"\nSelected bandwidth: {h_best:g} km")

    # --------------------------------------------------------
    # 2. INDEPENDENT HELD-OUT DRIFTER TEST
    # --------------------------------------------------------
    test_errors = kernel_evaluate(
        test, train, (h_best,),
        leave_one_out=False
    )

    if test_errors.empty:
        test_scores = pd.DataFrame()
        print("No comparable held-out test cells.")
    else:
        test_scores = kernel_score_table(test_errors)

        print("\nINDEPENDENT HELD-OUT TEST:")
        print(test_scores.round(4).to_string(index=False))

        for scope in SEASONS + ["Overall"]:
            s = test_scores[test_scores["scope"] == scope]

            if len(s) == 2:
                raw = float(s.loc[
                    s["bandwidth_km"] == 0,
                    "rmse_vector_mps"
                ].iloc[0])

                smooth = float(s.loc[
                    s["bandwidth_km"] == h_best,
                    "rmse_vector_mps"
                ].iloc[0])

                improvement = (
                    100 * (raw - smooth) / raw
                    if raw > 0 else np.nan
                )

                print(
                    f"{scope}: original={raw:.4f}, "
                    f"smoothed={smooth:.4f} m/s, "
                    f"improvement={improvement:+.1f}%"
                )

    save_table(test_scores, "kernel_heldout_test.csv")


    # --------------------------------------------------------
    # DRIFTER-LEVEL BOOTSTRAP: HELD-OUT RMSE IMPROVEMENT
    # --------------------------------------------------------
    if not test_errors.empty:
        save_table(
            test_errors,
            "kernel_heldout_drifter_errors.csv"
        )

        paired = test_errors.pivot(
            index=["drifter_id", "monsoon"],
            columns="bandwidth_km",
            values="mean_sq_vector_error"
        )[[0.0, h_best]].dropna()

        # Resample whole drifters, retaining both seasonal records.
        grouped = paired.groupby(level="drifter_id")
        group_sum = grouped.sum().to_numpy()
        group_n = grouped.size().to_numpy()

        rng_boot = np.random.default_rng(3001)
        n_boot_kernel = 5000

        indices = rng_boot.integers(
            0, len(group_sum),
            size=(n_boot_kernel, len(group_sum))
        )

        boot_mse = (
            group_sum[indices].sum(axis=1)
            / group_n[indices].sum(axis=1)[:, None]
        )

        raw_rmse = np.sqrt(boot_mse[:, 0])
        smooth_rmse = np.sqrt(boot_mse[:, 1])

        improvement = 100 * (
            raw_rmse - smooth_rmse
        ) / raw_rmse

        ci_low, ci_high = np.percentile(
            improvement, [2.5, 97.5]
        )

        positive_pct = 100 * np.mean(improvement > 0)

        overall_scores = test_scores[
            test_scores["scope"] == "Overall"
        ].set_index("bandwidth_km")

        raw_point = float(
            overall_scores.loc[0.0, "rmse_vector_mps"]
        )
        smooth_point = float(
            overall_scores.loc[h_best, "rmse_vector_mps"]
        )

        point_improvement = (
            100 * (raw_point - smooth_point) / raw_point
        )

        bootstrap_summary = pd.DataFrame([{
            "bandwidth_km": h_best,
            "n_test_drifters": len(group_sum),
            "n_bootstrap": n_boot_kernel,
            "observed_improvement_pct": point_improvement,
            "bootstrap_mean_improvement_pct": improvement.mean(),
            "ci95_low_pct": ci_low,
            "ci95_high_pct": ci_high,
            "bootstrap_positive_pct": positive_pct
        }])

        save_table(
            bootstrap_summary,
            "kernel_heldout_bootstrap.csv"
        )

        print("\nKERNEL BOOTSTRAP ROBUSTNESS:")
        print(bootstrap_summary.round(3).to_string(index=False))


    # --------------------------------------------------------
    # 3. FIT FINAL SMOOTHED MAP USING ALL DRIFTERS
    # --------------------------------------------------------
    smoothed = grid.copy()

    kernel_columns = [
        "kernel_ve", "kernel_vn", "kernel_speed",
        "kernel_direction_deg", "kernel_vector_change_mps",
        "kernel_direction_change_deg"
    ]

    for col in kernel_columns:
        smoothed[col] = np.nan

    for season in SEASONS:
        indices = smoothed.index[
            (smoothed["monsoon"] == season)
            & smoothed["sufficient"]
        ]

        source = smoothed.loc[indices]
        pred = kernel_predict(source, source, h_best)

        smoothed.loc[indices, "kernel_ve"] = pred[:, 0]
        smoothed.loc[indices, "kernel_vn"] = pred[:, 1]

        smoothed.loc[indices, "kernel_speed"] = np.hypot(
            pred[:, 0], pred[:, 1]
        )

        smoothed.loc[
            indices, "kernel_direction_deg"
        ] = np.degrees(np.arctan2(pred[:, 1], pred[:, 0]))

        smoothed.loc[
            indices, "kernel_vector_change_mps"
        ] = np.hypot(
            source["ve"].to_numpy() - pred[:, 0],
            source["vn"].to_numpy() - pred[:, 1]
        )

        smoothed.loc[
            indices, "kernel_direction_change_deg"
        ] = angle_between(
            source["ve"], source["vn"],
            pred[:, 0], pred[:, 1]
        )

    # Original coverage mask must remain unchanged.
    assert smoothed.loc[
        ~smoothed["sufficient"], "kernel_ve"
    ].isna().all()

    assert int(smoothed["kernel_ve"].notna().sum()) == int(
        smoothed["sufficient"].sum()
    )

    save_table(
        smoothed,
        f"kernel_smoothed_grid_{ACTIVE}.csv"
    )

    # --------------------------------------------------------
    # 4. FIGURE — ORIGINAL VS SMOOTHED
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        2, 2, figsize=(17, 13),
        constrained_layout=True
    )

    max_speed = max(
        float(smoothed["speed"].max()),
        float(smoothed["kernel_speed"].max())
    )

    norm = Normalize(0, max_speed)
    cmap = plt.get_cmap("YlGnBu")

    for row, season in enumerate(SEASONS):
        raw = smoothed[
            smoothed["monsoon"] == season
        ]

        season_cell_map(
            axes[row, 0], raw,
            "speed", cmap, norm,
            f"{season}: original drifter-weighted",
            arrows=True
        )

        new = raw.copy()
        new["ve"] = new["kernel_ve"]
        new["vn"] = new["kernel_vn"]
        new["speed"] = new["kernel_speed"]

        season_cell_map(
            axes[row, 1], new,
            "speed", cmap, norm,
            f"{season}: Gaussian kernel (h={h_best:g} km)",
            arrows=True
        )

    add_land_legend(
        axes[0, 0], [INSUFFICIENT_PATCH]
    )

    add_colorbar(
        fig, axes, cmap, norm,
        "Mean vector speed (m/s)"
    )

    fig.suptitle(
        f"Original vs kernel-smoothed circulation ({dataset_label()})",
        fontsize=15
    )

    save_fig(fig, "13_kernel_raw_vs_smoothed.png", dpi=250)

    # --------------------------------------------------------
    # 5. FIGURE — BANDWIDTH CROSS-VALIDATION
    # --------------------------------------------------------
    fig, ax = plt.subplots(
        figsize=(9, 6), constrained_layout=True
    )

    for season, marker in [
        ("Southwest", "o"),
        ("Northeast", "s"),
        ("Overall", "^")
    ]:
        s = cv_scores[
            cv_scores["scope"] == season
        ].sort_values("bandwidth_km")

        if not s.empty:
            ax.plot(
                s["bandwidth_km"],
                s["rmse_vector_mps"],
                marker=marker,
                label=season
            )

    ax.axvline(
        h_best,
        color="grey",
        linestyle="--",
        label=f"Selected h = {h_best:g} km"
    )

    ax.set(
        xlabel="Bandwidth h (km); 0 = original grid",
        ylabel="Drifter-grouped vector RMSE (m/s)",
        title="Training cross-validation for bandwidth selection"
    )

    ax.legend()
    ax.grid(alpha=0.2)

    save_fig(fig, "14_kernel_bandwidth_cv.png")

    # --------------------------------------------------------
    # 6. FIGURE — MAGNITUDE OF SMOOTHING DIFFERENCES
    # --------------------------------------------------------
    fig, axes = plt.subplots(
        1, 2, figsize=(16, 7),
        constrained_layout=True
    )

    max_diff = float(
        smoothed["kernel_vector_change_mps"].max()
    )

    norm_diff = Normalize(0, max(max_diff, 1e-8))
    diff_cmap = plt.get_cmap("Oranges")

    for ax, season in zip(axes, SEASONS):
        g = smoothed[
            smoothed["monsoon"] == season
        ]

        season_cell_map(
            ax, g,
            "kernel_vector_change_mps",
            diff_cmap, norm_diff,
            f"{season}: magnitude of smoothing change"
        )

    add_land_legend(axes[0], [INSUFFICIENT_PATCH])

    add_colorbar(
        fig, axes, diff_cmap, norm_diff,
        "Vector change (m/s)"
    )

    save_fig(fig, "15_kernel_difference.png")

    # --------------------------------------------------------
    # 7. WRITE INTERPRETATION SUMMARY
    # --------------------------------------------------------
    summary_path = os.path.join(
        OUT_DIR, "kernel_summary.txt"
    )

    with open(
        summary_path, "w", encoding="utf-8"
    ) as f:
        f.write("SPATIAL GAUSSIAN KERNEL SMOOTHING\n")
        f.write("Main climatology results unchanged.\n")
        f.write(
            "Only originally sufficient grid cells are smoothed.\n"
        )
        f.write(f"Selected bandwidth: {h_best:g} km\n")
        f.write(
            f"Training drifters: {len(train_ids)}\n"
        )
        f.write(
            f"Test drifters: {len(test_ids)}\n"
        )

        if not test_scores.empty:
            s = test_scores[
                test_scores["scope"] == "Overall"
            ]

            raw = float(s.loc[
                s["bandwidth_km"] == 0,
                "rmse_vector_mps"
            ].iloc[0])

            smooth = float(s.loc[
                s["bandwidth_km"] == h_best,
                "rmse_vector_mps"
            ].iloc[0])

            f.write(
                f"Original held-out RMSE: {raw:.5f} m/s\n"
            )
            f.write(
                f"Smoothed held-out RMSE: {smooth:.5f} m/s\n"
            )

            if raw > 0:
                f.write(
                    f"Improvement: {100 * (raw - smooth) / raw:+.2f}%\n"
                )
        else:
            f.write(
                "Insufficient independent test coverage.\n"
            )

        f.write(
            "Limitation: geographical smoothing does not "
            "account for land barriers or sharp ocean fronts.\n"
        )
        f.write(
            "Original bootstrap uncertainty does not describe "
            "the smoothed estimator.\n"
        )

    print("  -> kernel_summary.txt")
    return h_best


# ============================================================
# Gaussian process regression (kriging)
# ============================================================
# A statistical alternative to the gridded and kernel-smoothed maps. Each
# drifter's mean velocity in a cell is treated as a noisy measurement of a
# smooth underlying current field. The model has three parameters, learned
# from the data by maximum likelihood (no hand tuning):
#   - signal variance: how far the current typically departs from the
#     Bay-wide average,
#   - length scale (km): how far apart two places can be and still have
#     similar currents,
#   - noise variance: how much drifters passing through the same cell disagree
#     (eddies, different years, measurement error).
# The fitted model gives, for every cell, a best estimate of the mean current
# and an uncertainty that grows where drifters are scarce. Because the estimate
# at any place borrows information from nearby drifters, the map covers every
# ocean cell in the study region: cells with too few drifters for the main map,
# and also cells no drifter ever entered in that season. Such cells are marked
# on the maps, and their uncertainty is larger the further they are from data.
# This is the method geostatisticians call kriging. We use ordinary kriging:
# the Bay-wide mean current is treated as unknown and estimated together with
# the map (the kriging weights sum to one), and the uncertainty includes the
# extra error from estimating that mean.
#
# The correlation between places is modelled with a Matern (3/2) function, a
# standard choice for ocean currents: smooth, but rougher than a Gaussian bump.
# The noise variance plays the role of the "nugget". Eastward and northward
# velocity are modelled separately, for each season. As a check, the fitted
# model is drawn over the empirical variogram (half the average squared
# difference between drifter means, by distance) in figure 22.
#
# Reference: L. Astfalck, "Kriging demystified" (UNSW, 2026), whose recommended
# workflow this follows: ordinary kriging, a Matern covariance with a nugget,
# and the empirical variogram as the check of the fitted model. The parameters
# are fitted by maximum likelihood rather than by curve-fitting the variogram,
# because likelihood uses every pair of records rather than binned averages.

GP_LENGTH_BOUNDS_KM = (20.0, 1000.0)   # range searched for the length scale
GP_FIT_MAX_RECORDS = 1500              # records used to learn the three parameters (a random
                                       # subset if there are more, to keep fitting fast; all
                                       # records are always used for the predictions)
GP_REF_LAT = sum(LAT_RANGE) / 2        # latitude at which longitude degrees are converted to km


def gp_xy_km(lon, lat):
    """Positions in kilometres on a flat map of the Bay (error of a few
    percent at this scale, small compared with the length scales involved)."""
    lon, lat = np.asarray(lon, dtype=float), np.asarray(lat, dtype=float)
    x = 6371.0 * np.radians(lon - LON_RANGE[0]) * np.cos(np.radians(GP_REF_LAT))
    y = 6371.0 * np.radians(lat - LAT_RANGE[0])
    return np.column_stack([x, y])


def matern32(distance_km, variance, length_km):
    """Matern 3/2 covariance: how strongly the current at two places is related,
    as a function of the distance between them."""
    s = np.sqrt(3.0) * distance_km / length_km
    return variance * (1.0 + s) * np.exp(-s)


def gp_neg_log_likelihood(log_params, distance_km, y):
    """How poorly a set of parameters explains the data (lower is better)."""
    variance, length_km, noise = np.exp(log_params)
    K = matern32(distance_km, variance, length_km) + noise * np.eye(len(y))
    try:
        L = cholesky(K, lower=True)
    except np.linalg.LinAlgError:
        return 1e10
    alpha = cho_solve((L, True), y)
    return 0.5 * y @ alpha + np.log(np.diag(L)).sum() + 0.5 * len(y) * np.log(2 * np.pi)


def gp_fit(xy, y, rng):
    """Learn the signal variance, length scale and noise variance by maximum
    likelihood. Returns them with the Bay-wide mean the field varies around."""
    y = np.asarray(y, dtype=float)
    mean = y.mean()
    if len(y) > GP_FIT_MAX_RECORDS:
        pick = rng.choice(len(y), GP_FIT_MAX_RECORDS, replace=False)
        xy, y = xy[pick], y[pick]
    resid = y - mean
    var0 = resid.var()
    distance = cdist(xy, xy)
    start = np.log([var0 / 2, 150.0, var0 / 2])
    bounds = [(np.log(var0 * 1e-3), np.log(var0 * 10)), tuple(np.log(GP_LENGTH_BOUNDS_KM)),
              (np.log(var0 * 1e-3), np.log(var0 * 10))]
    res = minimize(gp_neg_log_likelihood, start, args=(distance, resid), method="L-BFGS-B", bounds=bounds)
    variance, length_km, noise = np.exp(res.x)
    return {"mean": mean, "variance": variance, "length_km": length_km, "noise": noise}


def gp_predict(xy_train, y_train, xy_new, p):
    """Best estimate and standard deviation of the underlying mean current at
    new positions, given the training records and fitted parameters `p`.
    The standard deviation describes the mean current itself; an individual
    drifter would scatter around it with the additional noise variance."""
    # Ordinary kriging: the mean is estimated from the data (generalised least
    # squares), which is the same as requiring the kriging weights to sum to one.
    y = np.asarray(y_train, dtype=float)
    K = matern32(cdist(xy_train, xy_train), p["variance"], p["length_km"]) + p["noise"] * np.eye(len(xy_train))
    L = cholesky(K, lower=True)
    K_new = matern32(cdist(xy_new, xy_train), p["variance"], p["length_km"])
    ones = np.ones(len(y))
    Kinv_1 = cho_solve((L, True), ones)
    mean = (Kinv_1 @ y) / (Kinv_1 @ ones)
    alpha = cho_solve((L, True), y - mean)
    v = solve_triangular(L, K_new.T, lower=True)
    # Simple-kriging variance plus the extra uncertainty from estimating the mean
    extra = (1.0 - K_new @ Kinv_1) ** 2 / (Kinv_1 @ ones)
    sd = np.sqrt(np.clip(p["variance"] - (v ** 2).sum(axis=0) + extra, 0.0, None))
    return mean + K_new @ alpha, sd


GP_VARIOGRAM_BIN_KM = 50.0     # width of the distance bins in the empirical variogram
GP_VARIOGRAM_MAX_KM = 1000.0   # largest distance shown


def gp_variogram_figure(rec, fitted, rng):
    """Empirical variogram of the drifter means against the fitted model, for
    each season and velocity component (figure 22). gamma(h) is half the
    average squared difference between two records h km apart; the fitted
    model is nugget + signal variance x (1 - Matern correlation)."""
    edges = np.arange(0.0, GP_VARIOGRAM_MAX_KM + GP_VARIOGRAM_BIN_KM, GP_VARIOGRAM_BIN_KM)
    fig, axes = plt.subplots(len(fitted), 2, figsize=(12, 4.2 * len(fitted)), squeeze=False,
                             constrained_layout=True)
    rows = []
    for i, (season, comp) in enumerate(fitted.items()):
        r = rec[rec["monsoon"] == season]
        if len(r) > GP_FIT_MAX_RECORDS:
            r = r.iloc[rng.choice(len(r), GP_FIT_MAX_RECORDS, replace=False)]
        d = cdist(*(2 * [gp_xy_km(r["lon0"] + GRID / 2, r["lat0"] + GRID / 2)]))
        iu = np.triu_indices(len(r), k=1)
        d = d[iu]
        for j, c in enumerate(("ve", "vn")):
            y = r[c].to_numpy()
            half_sq = 0.5 * (y[iu[0]] - y[iu[1]]) ** 2
            b = np.digitize(d, edges) - 1
            ok = (b >= 0) & (b < len(edges) - 1)
            gamma = np.bincount(b[ok], half_sq[ok], len(edges) - 1)
            n = np.bincount(b[ok], minlength=len(edges) - 1)
            centre = (edges[:-1] + edges[1:]) / 2
            show = n >= 30
            p = comp[c]
            h = np.linspace(1.0, GP_VARIOGRAM_MAX_KM, 300)
            model = p["noise"] + p["variance"] - matern32(h, p["variance"], p["length_km"])
            ax = axes[i, j]
            ax.plot(centre[show], gamma[show] / np.maximum(n[show], 1), "o", color="tab:blue",
                    label="Empirical variogram")
            ax.plot(h, model, color="black", label="Fitted Matern 3/2 model")
            ax.axhline(p["noise"], color="grey", linestyle=":", label="Nugget (noise variance)")
            ax.set_title(f"{season} monsoon, {'eastward' if c == 've' else 'northward'} velocity")
            ax.set_xlabel("Distance between records (km)")
            ax.set_ylabel("Semivariance (m²/s²)")
            ax.set_ylim(bottom=0)
            ax.grid(alpha=0.3)
            rows += [{"monsoon": season, "component": c, "bin_centre_km": x, "n_pairs": k,
                      "semivariance": g / k if k else np.nan} for x, k, g in zip(centre, n, gamma)]
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(f"Empirical variogram and fitted kriging model ({dataset_label()})", fontsize=14)
    save_fig(fig, "22_gp_variogram.png")
    save_table(pd.DataFrame(rows), "gp_variogram.csv")


GP_MIN_SEA_FRACTION = 0.10   # a cell is mapped by kriging if at least this share of it
                             # is sea; this covers coastal cells whose centre is on land,
                             # while leaving out cells that are almost entirely land


def ocean_cells():
    """Every grid cell in the study region that is at least GP_MIN_SEA_FRACTION
    sea, according to the Natural Earth coastline drawn on the maps.
    Returns None if the coastline is unavailable."""
    land = _get_land()
    if not land:
        return None
    from shapely.geometry import Polygon, box
    from shapely.ops import unary_union
    land_shape = unary_union([Polygon(p).buffer(0) for p in land if len(p) >= 4])
    lon0, lat0 = np.meshgrid(np.arange(LON_RANGE[0], LON_RANGE[1], GRID),
                             np.arange(LAT_RANGE[0], LAT_RANGE[1], GRID))
    cells = pd.DataFrame({"lon0": lon0.ravel(), "lat0": lat0.ravel()})
    sea = [1 - box(x, y, x + GRID, y + GRID).intersection(land_shape).area / GRID ** 2
           for x, y in zip(cells["lon0"], cells["lat0"])]
    cells["sea_fraction"] = np.round(sea, 3)
    return cells[cells["sea_fraction"] >= GP_MIN_SEA_FRACTION].reset_index(drop=True)


GP_FEW_STYLE = dict(marker="x", color="grey", s=14, zorder=3.5)
GP_NONE_STYLE = dict(marker="o", facecolors="none", edgecolors="dimgrey", s=16, linewidths=0.8, zorder=3.5)


def gp_mark_coverage(ax, cells):
    """Mark kriged cells that the main map would not show: cells with too few
    drifters (grey cross) and cells no drifter entered (open circle)."""
    few = cells[cells["sampled"] & ~cells["sufficient"]]
    none = cells[~cells["sampled"]]
    ax.scatter(few["lon"], few["lat"], **GP_FEW_STYLE)
    ax.scatter(none["lon"], none["lat"], **GP_NONE_STYLE)


def gp_coverage_legend():
    """Legend entries for the markers drawn by gp_mark_coverage."""
    return [plt.Line2D([], [], marker="x", color="grey", linestyle="",
                       label=f"Too few drifters for the main map (<{MIN_DRIFTERS})"),
            plt.Line2D([], [], marker="o", markerfacecolor="none", markeredgecolor="dimgrey",
                       linestyle="", label="No drifters: estimated from nearby cells only")]


def gp_analysis(per, grid, h_best):
    """Gaussian process (kriging) map of the mean current for the active
    dataset: parameters learned from the training drifters, tested on the
    held-out drifters against the raw grid and the kernel-smoothed map, then
    used to map every ocean cell in the study region with an uncertainty."""
    section(f"GAUSSIAN PROCESS REGRESSION / KRIGING ({dataset_label()})")
    rec = per[per["n_obs"] >= MIN_OBS_PER_DRIFTER]
    train_ids, test_ids = train_test_drifters(per)
    train = rec[rec["drifter_id"].isin(train_ids)]
    test = rec[rec["drifter_id"].isin(test_ids)]
    rng = np.random.default_rng(SEED)
    print(f"Drifters: {len(train_ids)} training, {len(test_ids)} testing (same split as the kernel smoothing)")

    # 1. Learn the parameters on the training drifters, and score each method
    #    on the test drifters, on exactly the same cells: cells the training
    #    drifters map in the raw grid (so all three methods can be compared).
    params, fitted, cases = [], {}, []
    for season in SEASONS:
        tr = train[train["monsoon"] == season]
        if len(tr) < 10:
            continue
        xy_tr = gp_xy_km(tr["lon0"] + GRID / 2, tr["lat0"] + GRID / 2)
        fitted[season] = {c: gp_fit(xy_tr, tr[c], rng) for c in ("ve", "vn")}
        for c, p in fitted[season].items():
            params.append({"monsoon": season, "component": c, "n_records": len(tr), "mean_mps": p["mean"],
                           "signal_sd_mps": np.sqrt(p["variance"]), "length_scale_km": p["length_km"],
                           "noise_sd_mps": np.sqrt(p["noise"])})

        ref = velocity_grid(per[(per["monsoon"] == season) & per["drifter_id"].isin(train_ids)])
        eligible = ref[ref["sufficient"]][["lon0", "lat0", "lon", "lat", "ve", "vn"]]
        c = test[test["monsoon"] == season].merge(eligible, on=["lon0", "lat0"], suffixes=("_true", "_raw"))
        if c.empty:
            continue
        xy_c = gp_xy_km(c["lon"], c["lat"])
        c["ve_gp"], c["sd_ve"] = gp_predict(xy_tr, tr["ve"], xy_c, fitted[season]["ve"])
        c["vn_gp"], c["sd_vn"] = gp_predict(xy_tr, tr["vn"], xy_c, fitted[season]["vn"])
        if h_best:
            k = kernel_predict(c[["lon", "lat"]], eligible, h_best)
            c["ve_kernel"], c["vn_kernel"] = k[:, 0], k[:, 1]
        # Does the 95% interval for an individual drifter contain the test value?
        # (uncertainty of the mean current + the drifter-to-drifter noise)
        c["inside_95"] = ((np.abs(c["ve_true"] - c["ve_gp"])
                           <= 1.96 * np.sqrt(c["sd_ve"] ** 2 + fitted[season]["ve"]["noise"]))
                          & (np.abs(c["vn_true"] - c["vn_gp"])
                             <= 1.96 * np.sqrt(c["sd_vn"] ** 2 + fitted[season]["vn"]["noise"])))
        cases.append(c)
    params = pd.DataFrame(params)
    print("\nLearned parameters (maximum likelihood, training drifters):")
    print(params.round(3).to_string(index=False))
    save_table(params, "gp_parameters.csv")
    if fitted:
        gp_variogram_figure(train, fitted, np.random.default_rng(SEED))
    if not cases:
        print("No comparable held-out test cells.")
        return None

    cases = pd.concat(cases, ignore_index=True)
    methods = {"raw grid": "raw", "Gaussian process": "gp"}
    if h_best:
        methods[f"kernel (h={h_best:g} km)"] = "kernel"
    for name, tag in methods.items():
        cases[name] = (cases["ve_true"] - cases[f"ve_{tag}"]) ** 2 + (cases["vn_true"] - cases[f"vn_{tag}"]) ** 2
    save_table(cases, "gp_heldout_cases.csv")

    # RMSE as in the kernel test: average error per held-out drifter-season first,
    # so each test drifter counts equally however many cells it crossed
    per_ds = cases.groupby(["monsoon", "drifter_id"])[list(methods)].mean()
    rows = []
    for scope in SEASONS + ["Overall"]:
        part = per_ds if scope == "Overall" else per_ds[per_ds.index.get_level_values("monsoon") == scope]
        if part.empty:
            continue
        for name in methods:
            rows.append({"scope": scope, "method": name, "rmse_vector_mps": np.sqrt(part[name].mean()),
                         "n_drifter_seasons": len(part)})
    scores = pd.DataFrame(rows)
    print("\nIndependent held-out test (lower RMSE is better):")
    print(scores.round(4).to_string(index=False))
    save_table(scores, "gp_heldout_test.csv")

    # Improvement over the raw grid, with a 95% CI from resampling test drifters
    g = per_ds.groupby(level="drifter_id")
    sums, counts = g.sum(), g.size().to_numpy()
    draws = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(sums), len(sums))
        mse = sums.iloc[pick].sum() / counts[pick].sum()
        draws.append(100 * (1 - np.sqrt(mse["Gaussian process"] / mse["raw grid"])))
    overall = scores[scores["scope"] == "Overall"].set_index("method")["rmse_vector_mps"]
    improvement = 100 * (1 - overall["Gaussian process"] / overall["raw grid"])
    ci = np.percentile(draws, [2.5, 97.5])
    coverage = cases["inside_95"].mean() * 100
    print(f"\nGaussian process vs raw grid: RMSE {overall['Gaussian process']:.4f} vs {overall['raw grid']:.4f} m/s, "
          f"reduction {improvement:+.1f}% (95% CI {ci[0]:+.1f} to {ci[1]:+.1f}%)")
    print(f"Calibration: {coverage:.1f}% of held-out drifter means fall inside the 95% interval "
          f"for both components (about 90% expected if the uncertainty is right)")
    save_table(pd.DataFrame([{"improvement_vs_raw_pct": improvement, "ci95_low_pct": ci[0],
                              "ci95_high_pct": ci[1], "coverage_95_pct": coverage,
                              "n_test_drifters": len(sums)}]), "gp_heldout_bootstrap.csv")

    # 2. Final map: the validated model applied to every drifter of the dataset,
    #    predicting every ocean cell in the study region, each with its own
    #    uncertainty. 'sampled' records whether any drifter entered the cell in
    #    that season; cells with no drifters are estimated from nearby cells only.
    targets = ocean_cells()
    if targets is None:
        print("\nCoastline unavailable: the kriging map covers only cells drifters entered.")
    parts = []
    for season, p in fitted.items():
        r = rec[rec["monsoon"] == season]
        xy_r = gp_xy_km(r["lon0"] + GRID / 2, r["lat0"] + GRID / 2)
        sampled = grid[grid["monsoon"] == season][["lon0", "lat0", "n_drifters", "sufficient", "confidence"]]
        if targets is None:
            g = sampled.copy()
        else:
            cells = pd.concat([targets[["lon0", "lat0"]], sampled[["lon0", "lat0"]]]).drop_duplicates()
            g = cells.merge(sampled, on=["lon0", "lat0"], how="left")
        g.insert(0, "monsoon", season)
        g["lon"], g["lat"] = g["lon0"] + GRID / 2, g["lat0"] + GRID / 2
        g["sampled"] = g["n_drifters"].notna()
        g["n_drifters"] = g["n_drifters"].fillna(0).astype(int)
        g["sufficient"] = g["sufficient"].fillna(False).astype(bool)
        g["confidence"] = g["confidence"].fillna("no drifters")
        xy_g = gp_xy_km(g["lon"], g["lat"])
        g["gp_ve"], g["gp_sd_ve"] = gp_predict(xy_r, r["ve"], xy_g, p["ve"])
        g["gp_vn"], g["gp_sd_vn"] = gp_predict(xy_r, r["vn"], xy_g, p["vn"])
        parts.append(g)
    gp = pd.concat(parts, ignore_index=True)
    gp["gp_speed"] = np.hypot(gp["gp_ve"], gp["gp_vn"])
    gp["gp_direction_deg"] = np.degrees(np.arctan2(gp["gp_vn"], gp["gp_ve"]))
    gp["gp_sd_vector"] = np.hypot(gp["gp_sd_ve"], gp["gp_sd_vn"])
    save_table(gp, f"gp_grid_{ACTIVE}.csv")

    print("\nCells on the kriging map:")
    for season in fitted:
        s = gp[gp["monsoon"] == season]
        sd = s.groupby(np.select([s["sufficient"], s["sampled"]], ["mapped", "few"], "none"))["gp_sd_vector"].median()
        print(f"  {season}: {s['sufficient'].sum()} also on the main map "
              f"(median SD {sd.get('mapped', np.nan):.3f} m/s), "
              f"{(s['sampled'] & ~s['sufficient']).sum()} with too few drifters "
              f"({sd.get('few', np.nan):.3f}), {(~s['sampled']).sum()} with no drifters "
              f"({sd.get('none', np.nan):.3f})")

    # Monsoon reversal on the Gaussian process map, over every cell drifters
    # entered in both seasons (cells with no drifters are left out, so the
    # figure is not driven by estimates made purely from neighbouring cells)
    entered = gp[gp["sampled"]]
    both = (entered[entered["monsoon"] == "Southwest"].set_index(["lon0", "lat0"])
            .join(entered[entered["monsoon"] == "Northeast"].set_index(["lon0", "lat0"]),
                  lsuffix="_sw", rsuffix="_ne", how="inner").reset_index())
    both["angle_change"] = angle_between(both["gp_ve_sw"], both["gp_vn_sw"], both["gp_ve_ne"], both["gp_vn_ne"])
    reversal = {}
    print("\nMonsoon reversal on the Gaussian process map (cells sampled in both seasons):")
    for region, mask in [("Whole Bay of Bengal", np.ones(len(both), bool)),
                         (f"Approx. EICC band {EICC_LON[0]:.0f}-{EICC_LON[1]:.0f}E", in_eicc(both["lon0"]))]:
        a = both.loc[mask, "angle_change"]
        reversal[region] = {"cells": len(a), "gt90": (a > 90).mean() * 100, "gt135": (a > 135).mean() * 100}
        print(f"  {region}: {len(a)} cells, > 90° {reversal[region]['gt90']:.1f}%, "
              f"> 135° {reversal[region]['gt135']:.1f}%")

    # Figures: mean current and its uncertainty for every ocean cell, with the
    # cells the main map would not show marked
    show = gp.assign(ve=gp["gp_ve"], vn=gp["gp_vn"], speed=gp["gp_speed"], sufficient=True)
    vmax = np.ceil(max(show["speed"].max(), grid.loc[grid["sufficient"], "speed"].max()) * 10) / 10
    norm, cmap = Normalize(0, vmax), plt.get_cmap("YlGnBu")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
    for ax, season in zip(axes, SEASONS):
        season_cell_map(ax, show[show["monsoon"] == season], "speed", cmap, norm, f"{season} monsoon", arrows=True)
        gp_mark_coverage(ax, gp[gp["monsoon"] == season])
    add_land_legend(axes[0], gp_coverage_legend())
    add_colorbar(fig, axes, cmap, norm, "Mean current speed (m/s)")
    fig.suptitle(f"Gaussian process (kriging) mean surface circulation ({dataset_label()})", fontsize=15)
    save_fig(fig, "20_gp_circulation.png", dpi=300)

    sd_show = gp.assign(sufficient=True)
    sd_norm, sd_cmap = Normalize(0, sd_show["gp_sd_vector"].quantile(0.95)), plt.get_cmap("Oranges")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
    for ax, season in zip(axes, SEASONS):
        season_cell_map(ax, sd_show[sd_show["monsoon"] == season], "gp_sd_vector", sd_cmap, sd_norm,
                        f"{season} monsoon")
        gp_mark_coverage(ax, gp[gp["monsoon"] == season])
    add_land_legend(axes[0], gp_coverage_legend())
    add_colorbar(fig, axes, sd_cmap, sd_norm, "Uncertainty of the mean current (m/s, vector SD)")
    fig.suptitle(f"Gaussian process uncertainty ({dataset_label()})", fontsize=15)
    save_fig(fig, "21_gp_uncertainty.png")

    return {"improvement": improvement, "ci": ci, "coverage": coverage, "rmse": overall,
            "params": params, "reversal": reversal}


# ============================================================
# Smooth temporal weighting (alternative to fixed seasons)
# ============================================================
# The main analysis uses fixed seasons: an observation in September counts
# fully towards the Southwest monsoon and one in October not at all. Here each
# calendar month gets its own map instead. Observations from every month are
# used, weighted by how close their month is to the month being mapped: the
# weight falls off as a Gaussian of the gap in months, measured around the
# calendar so that December sits next to January. Observations from all years
# are pooled, as in the main analysis.
#
# Drifters still count equally. Each drifter's observations in a cell are
# averaged using the month weights, and the drifter then contributes with the
# weight of its closest month, so a drifter seen in the target month counts
# fully and one seen only two months away counts much less. A cell is mapped
# when these weights add up to at least MIN_DRIFTERS, which is the same rule as
# the main analysis when only the target month is used.
#
# The width of the Gaussian (sigma) is chosen by leave-one-drifter-out
# cross-validation, and the result is compared with the fixed seasons on the
# same test cases.

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
FIXED_SEASONS = "fixed seasons (main analysis)"
TEMPORAL_KEY = ["target_month", "lon0", "lat0"]


def temporal_label(sigma):
    """Readable name for one temporal weighting width."""
    return "target month only" if sigma == 0 else f"Gaussian, sigma = {sigma:g} months"


def temporal_weights(sigma):
    """12 x 12 weight table: row = month being mapped, column = month the
    observation was made in. sigma = 0 uses the target month only."""
    months = np.arange(12)
    gap = np.abs(months[:, None] - months[None, :])
    gap = np.minimum(gap, 12 - gap)             # distance around the calendar
    if sigma == 0:
        return (gap == 0).astype(float)
    return np.exp(-0.5 * (gap / sigma) ** 2)


def fixed_season_weights():
    """The same table for the main analysis's fixed seasons: weight 1 for every
    month in the same monsoon as the target month and 0 otherwise. Rows for
    inter-monsoon months are all zero (those months are not mapped)."""
    season = [SEASON_OF_MONTH.get(m) for m in range(1, 13)]
    return np.array([[float(a is not None and a == b) for b in season] for a in season])


def monthly_drifter_means(obs):
    """Mean velocity of every drifter in every cell in each calendar month,
    keeping drifters with at least MIN_OBS_PER_DRIFTER hourly observations
    there. Every weighting scheme is built from these same records, so the
    comparison between schemes is like-for-like."""
    rec = (obs.groupby(["month", "lon0", "lat0", "drifter_id"])
              .agg(ve=("ve", "mean"), vn=("vn", "mean"), n_obs=("ve", "count"))
              .reset_index())
    return rec[rec["n_obs"] >= MIN_OBS_PER_DRIFTER].reset_index(drop=True)


def temporal_estimates(rec, weights):
    """Temporally weighted, drifter-weighted cell means for every calendar month.

    Returns one row per target month, cell and contributing drifter, holding
    the drifter's weight, the cell estimate from all drifters, and the cell
    estimate with that drifter left out (used for cross-validation)."""
    parts = []
    for target in range(1, 13):
        w = weights[target - 1, rec["month"].to_numpy() - 1]
        r = rec[w > 0].assign(w=w[w > 0])
        if r.empty:
            continue
        r = r.assign(wve=r["w"] * r["ve"], wvn=r["w"] * r["vn"])
        d = (r.groupby(["lon0", "lat0", "drifter_id"])
              .agg(sum_w=("w", "sum"), wve=("wve", "sum"), wvn=("wvn", "sum"), weight=("w", "max"))
              .reset_index())
        d["target_month"] = target
        parts.append(d)
    d = pd.concat(parts, ignore_index=True)

    # Each drifter's month-weighted mean, then the weighted average over drifters
    d["ve_d"] = d["wve"] / d["sum_w"]
    d["vn_d"] = d["wvn"] / d["sum_w"]
    d["w_ve"] = d["weight"] * d["ve_d"]
    d["w_vn"] = d["weight"] * d["vn_d"]
    d["in_month"] = d["weight"] == 1.0          # drifter was observed in the target month itself
    cell = d.groupby(TEMPORAL_KEY)
    total_w = cell["weight"].transform("sum")
    sum_ve = cell["w_ve"].transform("sum")
    sum_vn = cell["w_vn"].transform("sum")
    d["ve"] = sum_ve / total_w
    d["vn"] = sum_vn / total_w
    d["weighted_drifters"] = total_w
    d["drifters_in_month"] = cell["in_month"].transform("sum")

    # The same estimate with this drifter removed
    rest_w = total_w - d["weight"]
    with np.errstate(invalid="ignore", divide="ignore"):
        d["ve_loo"] = np.where(rest_w > 1e-12, (sum_ve - d["w_ve"]) / rest_w, np.nan)
        d["vn_loo"] = np.where(rest_w > 1e-12, (sum_vn - d["w_vn"]) / rest_w, np.nan)
    d["weight_loo"] = rest_w
    return d


def temporal_grid(est):
    """One row per month and cell from temporal_estimates, in the same layout as
    the seasonal grids so the existing map functions can draw it."""
    g = est.drop_duplicates(TEMPORAL_KEY)[
        TEMPORAL_KEY + ["ve", "vn", "weighted_drifters", "drifters_in_month"]
    ].rename(columns={"target_month": "month"}).reset_index(drop=True)
    g["speed"] = np.hypot(g["ve"], g["vn"])
    g["direction_deg"] = np.degrees(np.arctan2(g["vn"], g["ve"]))
    g["sufficient"] = g["weighted_drifters"] >= MIN_DRIFTERS
    g["lon"] = g["lon0"] + GRID / 2
    g["lat"] = g["lat0"] + GRID / 2
    return g.sort_values(["month", "lon0", "lat0"], ignore_index=True)


def temporal_cv(rec, schemes):
    """Leave-one-drifter-out test of each weighting scheme. For every drifter,
    month and cell where it was observed, the drifter is removed, the cell is
    estimated from the remaining drifters, and the estimate is compared with the
    drifter's own mean velocity there. Returns the squared vector error of each
    scheme per test case (NaN where the scheme cannot map the cell)."""
    truth = rec.rename(columns={"month": "target_month", "ve": "ve_true", "vn": "vn_true"})
    key = TEMPORAL_KEY + ["drifter_id"]
    errors = truth[key].copy()
    for label, weights in schemes.items():
        est = temporal_estimates(rec, weights)
        m = truth.merge(est[key + ["ve_loo", "vn_loo", "weight_loo"]], on=key, how="left")
        sq = (m["ve_true"] - m["ve_loo"]) ** 2 + (m["vn_true"] - m["vn_loo"]) ** 2
        errors[label] = np.where(m["weight_loo"] >= MIN_DRIFTERS, sq, np.nan)
    return errors


def temporal_score_table(errors, labels, months, scope):
    """RMSE of each scheme, using only test cases that every scheme in labels
    can map, so all schemes are scored on exactly the same cases."""
    sub = errors[errors["target_month"].isin(months)].dropna(subset=labels)
    return pd.DataFrame([{
        "scope": scope, "method": label,
        "rmse_vector_mps": np.sqrt(sub[label].mean()),
        "n_test_cases": len(sub), "n_drifters": sub["drifter_id"].nunique(),
    } for label in labels])


def bootstrap_improvement(errors, new, old, seed=SEED):
    """95% confidence interval for the % reduction in RMSE of scheme `new`
    relative to scheme `old`, resampling whole drifters."""
    per = errors.dropna(subset=[new, old]).groupby("drifter_id")[[new, old]].sum()
    new_sq, old_sq = per[new].to_numpy(), per[old].to_numpy()
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, len(per), len(per)), minlength=len(per))
        draws.append(100 * (1 - np.sqrt((w * new_sq).sum() / (w * old_sq).sum())))
    return np.percentile(draws, [2.5, 97.5])


def temporal_weighting_analysis(data_all_months):
    """Monthly circulation maps using smooth temporal weighting, with the
    weighting width chosen by cross-validation and tested against the fixed
    seasons. Uses the active dataset's observations from all twelve months."""
    section(f"SMOOTH TEMPORAL WEIGHTING ({dataset_label()}, all months)")
    rec = monthly_drifter_means(data_all_months)
    print(f"Drifter-month-cell records: {len(rec)} from {rec['drifter_id'].nunique()} drifters")

    # 1. Cross-validation of each weighting width, and of the fixed seasons
    sigma_labels = [temporal_label(s) for s in TEMPORAL_SIGMAS]
    schemes = {temporal_label(s): temporal_weights(s) for s in TEMPORAL_SIGMAS}
    schemes[FIXED_SEASONS] = fixed_season_weights()
    errors = temporal_cv(rec, schemes)
    season_months = sorted(SEASON_OF_MONTH)
    scores = pd.concat([
        temporal_score_table(errors, sigma_labels, range(1, 13), "All months"),
        temporal_score_table(errors, list(schemes), season_months, "SW + NE months"),
    ], ignore_index=True)
    print("\nLeave-one-drifter-out cross-validation (lower RMSE is better):")
    print(scores.round(4).to_string(index=False))
    save_table(scores, "temporal_weighting_cv.csv")

    # Width with the lowest error over all twelve months
    all_months = scores[scores["scope"] == "All months"].set_index("method")["rmse_vector_mps"]
    best_sigma = min(TEMPORAL_SIGMAS, key=lambda s: all_months[temporal_label(s)])
    best = temporal_label(best_sigma)
    print(f"\nSelected weighting: {best}")

    # Compare with the fixed seasons in the months both can map
    in_seasons = scores[scores["scope"] == "SW + NE months"].set_index("method")["rmse_vector_mps"]
    improvement = 100 * (1 - in_seasons[best] / in_seasons[FIXED_SEASONS])
    ci = bootstrap_improvement(errors[errors["target_month"].isin(season_months)], best, FIXED_SEASONS)
    print(f"SW + NE months: fixed seasons RMSE {in_seasons[FIXED_SEASONS]:.4f} m/s, "
          f"{best} RMSE {in_seasons[best]:.4f} m/s, "
          f"reduction {improvement:+.1f}% (95% CI {ci[0]:+.1f} to {ci[1]:+.1f}%)")
    print("(The width is chosen on the same drifters it is tested on, so this "
          "comparison is slightly optimistic for the temporal weighting.)")

    # 2. Monthly climatology with the selected width
    clim = temporal_grid(temporal_estimates(rec, temporal_weights(best_sigma)))
    save_table(clim, f"monthly_climatology_{ACTIVE}.csv")
    mapped = clim[clim["sufficient"]].groupby("month").size().reindex(range(1, 13), fill_value=0)
    print("\nMapped cells per month: " +
          ", ".join(f"{MONTH_NAMES[m - 1]} {n}" for m, n in mapped.items()))

    # 3. Maps for all twelve months, on one shared speed scale
    vmax = np.ceil(clim.loc[clim["sufficient"], "speed"].max() * 10) / 10
    norm, cmap = Normalize(0, vmax), plt.get_cmap("YlGnBu")
    fig, axes = plt.subplots(3, 4, figsize=(24, 17), constrained_layout=True)
    for ax, month in zip(axes.flat, range(1, 13)):
        season_cell_map(ax, clim[clim["month"] == month], "speed", cmap, norm,
                        f"{MONTH_NAMES[month - 1]} ({SEASON_OF_MONTH.get(month, 'Inter-monsoon')})",
                        arrows=True)
    add_land_legend(axes[0, 0], [Patch(**INSUFFICIENT_STYLE,
                                       label=f"Insufficient data (weighted drifters < {MIN_DRIFTERS})")],
                    fontsize=8)
    add_colorbar(fig, axes, cmap, norm, "Mean current speed (m/s)")
    fig.suptitle(f"Monthly surface circulation with smooth temporal weighting "
                 f"({best}, {dataset_label()})", fontsize=16)
    save_fig(fig, "16_monthly_circulation.png")

    # 4. Annual cycle of the northward current in the EICC band. Too few single
    # cells are sampled in every month, so the whole band is treated as one
    # region: the same weighting is applied to all drifter records in the band,
    # giving one drifter-weighted mean per month.
    band_rec = rec[in_eicc(rec["lon0"])].assign(lon0=EICC_LON[0], lat0=LAT_RANGE[0])

    def band_cycle(weights):
        g = temporal_grid(temporal_estimates(band_rec, weights)).set_index("month")
        return g["vn"].where(g["sufficient"]), g["weighted_drifters"]

    smooth_vn, smooth_n = band_cycle(temporal_weights(best_sigma))
    fixed_vn, _ = band_cycle(fixed_season_weights())
    cycle = pd.DataFrame({"smooth_vn": smooth_vn, "weighted_drifters": smooth_n,
                          "fixed_vn": fixed_vn}).reindex(range(1, 13))
    cycle.index.name = "month"
    save_table(cycle, "eicc_monthly_cycle.csv", index=True)
    print(f"\nEICC band {EICC_LON[0]:g}-{EICC_LON[1]:g}°E, mean northward velocity (m/s) by month:")
    print(cycle.round(3).rename(index=dict(enumerate(MONTH_NAMES, 1))).T.to_string())

    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for months, colour, label in [((6, 9), "coral", "Southwest months"), ((12, 12), "steelblue", None),
                                  ((1, 2), "steelblue", "Northeast months")]:
        ax.axvspan(months[0] - 0.5, months[1] + 0.5, color=colour, alpha=0.15, label=label)
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.plot(cycle.index, cycle["smooth_vn"], "o-", color="black", label=f"Monthly ({best})")
    for span in [(6, 9), (12, 12), (1, 2)]:
        value = cycle.loc[span[0], "fixed_vn"]
        ax.hlines(value, span[0] - 0.5, span[1] + 0.5, color="firebrick", linewidth=2.5,
                  label=FIXED_SEASONS.capitalize() if span == (6, 9) else None)
    ax.set_xticks(range(1, 13), labels=MONTH_NAMES)
    ax.set(xlim=(0.5, 12.5), ylabel="Mean northward velocity (m/s)",
           title=f"Annual cycle of the northward current, EICC band {EICC_LON[0]:g}-{EICC_LON[1]:g}°E "
                 f"({dataset_label()})")
    ax.legend(loc="best", fontsize=9)
    save_fig(fig, "17_eicc_annual_cycle.png")

    return {"best": best, "improvement": improvement, "ci": ci, "rmse": in_seasons}


def run(dataset):
    """Run the full analysis, start to finish, on one dataset: "drogued",
    "undrogued" or "all". Results go to outputs/<dataset>/."""
    section(f"DATASET: {DATASETS[dataset]}")
    section("LOAD")
    ds = load_subset()
    print(f"Observations: {ds.sizes['obs']}, drifters: {ds.sizes['traj']}")

    section("DATA QUALITY (raw subset)")
    quality = data_quality_table(ds)
    print(quality.round(3))
    save_table(quality, "data_quality.csv", index=True)

    section("CLEANING")
    obs, exclusions = clean_obs(build_obs_table(ds))
    season_obs = obs[obs["monsoon"].isin(SEASONS)]     # Southwest and Northeast months only
    data = select_dataset(season_obs, dataset)         # the drifters this run analyses
    steps = [{"step": "inter-monsoon months (analysis only)", "removed": len(obs) - len(season_obs),
              "remaining": len(season_obs)}]
    if dataset != "all":
        other = "undrogued" if dataset == "drogued" else "drogued"
        steps.append({"step": f"{other} observations (not in this dataset)",
                      "removed": len(season_obs) - len(data), "remaining": len(data)})
    exclusions = pd.concat([exclusions, pd.DataFrame(steps)], ignore_index=True)
    print(exclusions.to_string(index=False))
    save_table(exclusions, "data_exclusions.csv")
    print(f"\nTime span: {obs['time'].min():%Y-%m-%d} to {obs['time'].max():%Y-%m-%d}")

    explore(ds, obs)

    # One mean per drifter, per cell, per season: the building block of every step below
    per = per_drifter_means(data)
    threshold_sensitivity(per)

    section(f"GRIDDED CLIMATOLOGY ({dataset_label()})")
    grid = add_bootstrap_se(velocity_grid(per), per)
    print("Cells by confidence tier")
    print(pd.crosstab(grid["monsoon"], grid["confidence"]))
    save_table(grid, f"climatology_grid_{dataset}.csv")

    # Spatial extensions: kernel smoothing, then the Gaussian process (kriging),
    # tested on the same held-out drifters
    h_best = kernel_smoothing_analysis(per, grid)
    gp = gp_analysis(per, grid, h_best)

    circulation_figures(grid)
    reversal = reversal_analysis(per, grid)
    baseline = interannual_baseline(data, per)
    sst_grid = sst_analysis(data)

    # Alternative to fixed seasons: monthly maps with smooth temporal weighting
    temporal = temporal_weighting_analysis(select_dataset(obs, dataset))

    # Headline numbers: a readable summary.txt, plus headline.csv (one row),
    # which the comparison step reads to set the datasets side by side
    kept = grid[grid["sufficient"]]
    sst_kept = sst_grid[sst_grid["sufficient"]]
    head = {"dataset": dataset, "observations": len(data), "drifters": data["drifter_id"].nunique()}
    lines = [
        f"HEADLINE NUMBERS ({dataset_label()})",
        f"Rule: {GRID:g} deg cells, >= {MIN_DRIFTERS} drifters each with >= "
        f"{MIN_OBS_PER_DRIFTER} hourly obs, drifter-weighted mean, {N_BOOT} whole-drifter bootstrap",
        "",
        f"Observations after cleaning: {len(obs)} ({obs['drifter_id'].nunique()} drifters), "
        f"{obs['time'].min():%Y-%m} to {obs['time'].max():%Y-%m}",
        f"SW + NE observations: {len(season_obs)}; in this dataset: {len(data)} "
        f"({len(data) / len(season_obs) * 100:.1f}%) from {head['drifters']} drifters",
        "",
        "Mapped cells:",
    ]
    for season in SEASONS:
        k = kept[kept["monsoon"] == season]
        n_high = int((k["confidence"] == "high").sum())
        n_sst = int((sst_kept["monsoon"] == season).sum())
        key = "sw" if season == "Southwest" else "ne"
        head.update({f"{key}_cells": len(k), f"{key}_high_conf_cells": n_high,
                     f"{key}_mean_sst_c": sst_kept.loc[sst_kept["monsoon"] == season, "sst"].mean()})
        lines.append(f"  {season}: {len(k)} cells ({n_high} high confidence), SST cells: {n_sst}")
    lines += ["", "SW vs NE reversal (> 90 deg):"]
    for _, r in reversal.iterrows():
        key = "eicc" if r["region"].startswith("Approx. EICC") else "bay"
        head.update({f"{key}_cells_both": r["cells_both_seasons"],
                     f"{key}_reversed_pct": r["reversed_pct"],
                     f"{key}_reversed_ci95_low": r["ci95_low_pct"],
                     f"{key}_reversed_ci95_high": r["ci95_high_pct"],
                     f"{key}_gt135_pct": r["gt135_pct"],
                     f"{key}_gt135_ci95_low": r["gt135_ci95_low_pct"],
                     f"{key}_gt135_ci95_high": r["gt135_ci95_high_pct"]})
        lines.append(f"  {r['region']}: {r['reversed_pct']:.1f}% of "
                     f"{r['cells_both_seasons']} cells (bootstrap median {r['boot_median_pct']:.1f}%, "
                     f"95% CI {r['ci95_low_pct']:.1f}-{r['ci95_high_pct']:.1f}%); "
                     f"> 135 deg: {r['gt135_pct']:.1f}% (median {r['gt135_boot_median_pct']:.1f}%, "
                     f"95% CI {r['gt135_ci95_low_pct']:.1f}-{r['gt135_ci95_high_pct']:.1f}%)")
    lines += ["", "Interannual baseline (random halves of the years, % of cells changing direction):"]
    for _, t in baseline[baseline["split_by"] == "years"].iterrows():
        key = "eicc" if t["region"].startswith("Approx. EICC") else "bay"
        head[f"{key}_monsoon_share_gt{t['threshold_deg']}_pct"] = t["monsoon_share_pct"]
        lines.append(f"  > {t['threshold_deg']} deg, {t['region']}: SW vs NE {t['sw_vs_ne_pct_median']:.1f}%, SW vs SW "
                     f"{t['sw_vs_sw_pct_median']:.1f}%, NE vs NE {t['ne_vs_ne_pct_median']:.1f}%; monsoon excess "
                     f"{t['monsoon_excess_pct_median']:.1f} points (95% range {t['monsoon_excess_pct_ci95_low']:.1f}"
                     f" to {t['monsoon_excess_pct_ci95_high']:.1f}), {t['monsoon_share_pct']:.0f}% of the reversal")
    head.update({"temporal_best": temporal["best"], "temporal_rmse_reduction_pct": temporal["improvement"]})
    lines += ["", "Smooth temporal weighting vs fixed seasons (leave-one-drifter-out, SW + NE months):",
              f"  Selected: {temporal['best']}; RMSE {temporal['rmse'][temporal['best']]:.4f} m/s vs "
              f"{temporal['rmse'][FIXED_SEASONS]:.4f} m/s for fixed seasons, reduction "
              f"{temporal['improvement']:+.1f}% (95% CI {temporal['ci'][0]:+.1f} to {temporal['ci'][1]:+.1f}%)"]
    if gp is not None:
        bay = gp["reversal"]["Whole Bay of Bengal"]
        head.update({"gp_rmse_reduction_pct": gp["improvement"], "gp_rmse_reduction_ci95_low": gp["ci"][0],
                     "gp_rmse_reduction_ci95_high": gp["ci"][1], "gp_coverage_95_pct": gp["coverage"],
                     "gp_bay_cells_both": bay["cells"], "gp_bay_reversed_pct": bay["gt90"],
                     "gp_bay_gt135_pct": bay["gt135"]})
        lengths = ", ".join(f"{r.monsoon} {r.component} {r.length_scale_km:.0f} km"
                            for r in gp["params"].itertuples())
        lines += ["", "Gaussian process (kriging) vs raw grid (held-out test drifters):",
                  f"  RMSE {gp['rmse']['Gaussian process']:.4f} m/s vs {gp['rmse']['raw grid']:.4f} m/s, reduction "
                  f"{gp['improvement']:+.1f}% (95% CI {gp['ci'][0]:+.1f} to {gp['ci'][1]:+.1f}%); "
                  f"{gp['coverage']:.0f}% of test values inside the 95% interval",
                  f"  Length scales: {lengths}",
                  f"  Reversal on the GP map: > 90 deg in {bay['gt90']:.1f}% of {bay['cells']} cells, "
                  f"> 135 deg in {bay['gt135']:.1f}%"]
    save_table(pd.DataFrame([head]), "headline.csv")
    text = "\n".join(lines)
    section("SUMMARY")
    print(text)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nAll outputs written to {OUT_DIR}")


# ============================================================
# Comparison of the three datasets
# ============================================================
# Run after the datasets have been analysed. It reads each dataset's saved
# results from outputs/<dataset>/ and sets them side by side in
# outputs/comparison/: headline numbers, reversal percentages with confidence
# intervals, the circulation maps, and a cell-by-cell comparison of every pair
# of datasets. Datasets that have not been run yet are skipped.

def load_saved(name, table):
    """A table saved by an earlier run of dataset `name`, or None if missing."""
    path = os.path.join(OUTPUT_ROOT, name, "tables", table)
    return pd.read_csv(path) if os.path.exists(path) else None


def compare_datasets():
    """Side-by-side comparison of the drogued, undrogued and all-drifter results."""
    section("DATASET COMPARISON")
    grids = {name: load_saved(name, f"climatology_grid_{name}.csv") for name in DATASETS}
    ready = [name for name, g in grids.items() if g is not None]
    missing = [name for name in DATASETS if name not in ready]
    if missing:
        print("Not yet run (skipped): " + ", ".join(missing)
              + ". Run them with --dataset <name>, or all at once with --dataset every.")
    if not ready:
        print("No saved results to compare.")
        return
    print("Compared: " + ", ".join(DATASETS[name] for name in ready))

    # 1. Headline numbers, one column per dataset
    head = pd.concat([load_saved(name, "headline.csv") for name in ready], ignore_index=True)
    kernel = [load_saved(name, "kernel_heldout_bootstrap.csv") for name in ready]
    head["kernel_rmse_reduction_pct"] = [k["observed_improvement_pct"].iloc[0] if k is not None else np.nan
                                         for k in kernel]
    head = head.set_index("dataset")
    print("\nHeadline numbers by dataset:")
    print(head.T.to_string())
    save_table(head.T, "dataset_comparison.csv", index=True)

    # 2. Reversal percentages with confidence intervals
    reversal = pd.concat([load_saved(name, "reversal_summary.csv") for name in ready], ignore_index=True)
    save_table(reversal, "reversal_by_dataset.csv")
    regions = list(dict.fromkeys(reversal["region"]))
    colours = {"drogued": "steelblue", "undrogued": "darkorange", "all": "grey"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True, constrained_layout=True)
    for ax, (threshold, pct, lo, hi) in zip(axes, [
            (90, "reversed_pct", "ci95_low_pct", "ci95_high_pct"),
            (135, "gt135_pct", "gt135_ci95_low_pct", "gt135_ci95_high_pct")]):
        width = 0.8 / len(ready)
        for i, name in enumerate(ready):
            r = reversal[reversal["dataset"] == name].set_index("region").reindex(regions)
            x = np.arange(len(regions)) + (i - (len(ready) - 1) / 2) * width
            ax.bar(x, r[pct], width, color=colours.get(name, "grey"), alpha=0.8, label=DATASETS[name])
            # Error bars span the bootstrap 95% CI (clipped at the bar, since the
            # observed value can sit just outside the bootstrap interval)
            ax.errorbar(x, r[pct], yerr=[(r[pct] - r[lo]).clip(lower=0), (r[hi] - r[pct]).clip(lower=0)],
                        fmt="none", ecolor="black", capsize=4)
        chance = 100 * (180 - threshold) / 180
        ax.axhline(chance, color="grey", linestyle=":",
                   label="Expected if directions were random (50% for > 90°, 25% for > 135°)")
        ax.set_xticks(range(len(regions)), labels=regions)
        ax.set(title=f"Cells changing direction by > {threshold}°", ylim=(0, 100))
    axes[0].set_ylabel("% of cells (bar = observed, line = 95% CI)")
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc="outside lower center", ncol=len(names), fontsize=10)   # below the panels,
                                                                                         # clear of the error bars
    fig.suptitle("Monsoon reversal by dataset", fontsize=15)
    save_fig(fig, "19_reversal_by_dataset.png")

    # 3. Circulation maps: one row per dataset, on one shared speed scale
    vmax = np.ceil(max(grids[n].loc[grids[n]["sufficient"], "speed"].max() for n in ready) * 10) / 10
    norm, cmap = Normalize(0, vmax), plt.get_cmap("YlGnBu")
    fig, axes = plt.subplots(len(ready), 2, figsize=(16, 7 * len(ready)),
                             constrained_layout=True, squeeze=False)
    for row, name in enumerate(ready):
        g = grids[name]
        for ax, season in zip(axes[row], SEASONS):
            season_cell_map(ax, g[g["monsoon"] == season], "speed", cmap, norm,
                            f"{DATASETS[name].capitalize()}: {season}", arrows=True)
    add_land_legend(axes[0, 0], [INSUFFICIENT_PATCH])
    add_colorbar(fig, axes, cmap, norm, "Mean current speed (m/s)")
    fig.suptitle("Mean surface circulation by dataset", fontsize=15)
    save_fig(fig, "07_circulation_by_dataset.png")

    # 4. Cell-by-cell comparison of each pair (the first named is the reference)
    for i, a in enumerate(ready):
        for b in ready[i + 1:]:
            compare_pair(grids[a], grids[b], a, b)


def run_logged(name, task):
    """Point outputs at outputs/<name>/ and run `task`, saving everything it
    prints to report.txt in that folder as well as showing it on screen."""
    set_dataset(name)
    with open(os.path.join(OUT_DIR, "report.txt"), "w", encoding="utf-8") as log:
        sys.stdout = Tee(sys.__stdout__, log)
        try:
            task()
        finally:
            sys.stdout = sys.__stdout__


def main():
    """Read the command-line options and run the chosen dataset(s)."""
    parser = argparse.ArgumentParser(description="Bay of Bengal monsoon surface circulation from drifters.")
    parser.add_argument("--dataset", choices=list(DATASETS) + ["every"], default=DATASET,
                        help=f"dataset to analyse (default: {DATASET}); 'every' runs all three, "
                             "then the comparison")
    parser.add_argument("--compare", action="store_true",
                        help="only build the comparison, from results already saved")
    args = parser.parse_args()

    if args.compare:
        run_logged(COMPARISON, compare_datasets)
        return
    names = list(DATASETS) if args.dataset == "every" else [args.dataset]
    for name in names:
        run_logged(name, lambda: run(name))
    if args.dataset == "every":
        run_logged(COMPARISON, compare_datasets)


if __name__ == "__main__":
    main()
