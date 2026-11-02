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
2. Keep drogued drifters only for the main analysis (undrogued drifters are
   pushed by the wind). The analysis is repeated with all drifters as a
   robustness check.
3. Grid the observations into 1° cells. A cell is mapped only if at least
   3 different drifters each spent at least 12 hours in it.
4. Average each drifter's mean velocity in a cell, so that every drifter counts
   equally. This stops a single drifter caught in an eddy from dominating.
5. Estimate uncertainty by resampling whole drifters (bootstrap). Cells with
   10 or more drifters are labelled high confidence, 3-9 low confidence.
6. Report the percentage of cells where the current direction changes by more
   than 90° between monsoons, for the whole Bay and for the western Bay
   (80-85°E, an approximate East India Coastal Current band), with 95%
   confidence intervals.
7. As an alternative to the fixed seasons, build a map for every calendar
   month in which observations from neighbouring months count with smoothly
   decreasing weight. The amount of smoothing is chosen by cross-validation
   and compared with the fixed seasons.

How to run
----------
    .venv313\Scripts\python.exe project3001.py

The first run downloads and subsets the global dataset (a few minutes) and
caches it to `bob_subset.nc`. Later runs take about a minute.

Outputs (written to `outputs/`)
-------------------------------
    figures/   maps and charts (PNG)
    tables/    gridded climatology and supporting tables (CSV)
    report.txt full printed log of the run
    summary.txt headline numbers
    reversal_caption.txt  ready-to-use caption for the reversal map
"""
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
N_BOOT = 500                     # number of bootstrap resamples
SEED = 0                         # random seed, so results are reproducible

# --- East India Coastal Current (EICC) band ---
EICC_LON = (80.0, 85.0)          # western Bay longitude band used as an approximate EICC region
                                 # (a simple longitude band, not a coastline-following mask)

# --- Smooth temporal weighting (alternative to the fixed seasons above) ---
TEMPORAL_SIGMAS = (0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)   # widths of the Gaussian month weighting
                                                          # to test, in months; 0 = target month only

# --- Threshold sensitivity test: combinations of mapping rules to compare ---
SENS_MIN_DRIFTERS = [1, 3, 5, 10]
SENS_MIN_OBS = [1, 6, 12, 24, 48]

# --- File locations ---
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "bob_subset.nc")   # local copy of the regional data;
                                                   # delete it if the region above changes
OUT_DIR = os.path.join(HERE, "outputs")
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
        rows.append((*key, bve.std(ddof=1), bvn.std(ddof=1), np.hypot(bve, bvn).std(ddof=1)))
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
    recalculated. As in the main estimate, a cell needs at least MIN_DRIFTERS
    different drifters in each season to count. With eicc_only=True only the
    EICC band is used.

    Returns one row per threshold: [2.5th percentile, median, 97.5th percentile]."""
    q = per[per["n_obs"] >= MIN_OBS_PER_DRIFTER]
    if eicc_only:
        q = q[in_eicc(q["lon0"])]
    ids, drifter_idx = np.unique(q["drifter_id"].to_numpy(), return_inverse=True)
    cells, cell_idx = np.unique(q[["lon0", "lat0"]].to_numpy(), axis=0, return_inverse=True)
    code = cell_idx.ravel() * 2 + (q["monsoon"].to_numpy() == "Northeast")   # one slot per (cell, season)
    n_slots = len(cells) * 2
    ve, vn = q["ve"].to_numpy(), q["vn"].to_numpy()

    rng = np.random.default_rng(seed)
    results = []
    for _ in range(N_BOOT):
        # w = how many times each drifter was drawn in this resample (0, 1, 2, ...);
        # cell means are then weighted averages using these counts
        w = np.bincount(rng.integers(0, len(ids), len(ids)), minlength=len(ids))[drifter_idx]
        w_sum = np.bincount(code, w, n_slots)
        n_distinct = np.bincount(code, w > 0, n_slots)
        with np.errstate(invalid="ignore", divide="ignore"):
            mve = (np.bincount(code, w * ve, n_slots) / w_sum).reshape(-1, 2)
            mvn = (np.bincount(code, w * vn, n_slots) / w_sum).reshape(-1, 2)
        both = (n_distinct >= MIN_DRIFTERS).reshape(-1, 2).all(axis=1)
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
    """Overview of the cleaned data (all months): when and where drifters
    sampled, the share that were drogued, and current speeds by drogue status."""
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
    # reason the main analysis uses drogued drifters only.
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
    section("THRESHOLD SENSITIVITY (drogued only)")
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


def circulation_figures(grid_dr, grid_all):
    """Maps of mean surface current for each monsoon (main result), the same
    maps with undrogued drifters included for comparison, and the uncertainty map."""
    section("CIRCULATION MAPS")
    # Same speed colour scale on every map so they can be compared directly
    vmax = np.ceil(pd.concat([grid_dr.loc[grid_dr["sufficient"], "speed"],
                              grid_all.loc[grid_all["sufficient"], "speed"]]).max() * 10) / 10
    norm, cmap = Normalize(0, vmax), plt.get_cmap("YlGnBu")

    # Main result: drogued drifters only
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
    for ax, season in zip(axes, SEASONS):
        season_cell_map(ax, grid_dr[grid_dr["monsoon"] == season], "speed", cmap, norm,
                        f"{season} monsoon", arrows=True)
    add_land_legend(axes[0], [INSUFFICIENT_PATCH])
    add_colorbar(fig, axes, cmap, norm, "Mean current speed (m/s)")
    fig.suptitle("Drifter-weighted mean surface circulation (drogued drifters only)", fontsize=15)
    save_fig(fig, "06_circulation_drogued.png", dpi=300)

    # Comparison: all drifters (top) vs drogued only (bottom)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    for row, (label, grid) in enumerate([("All data", grid_all), ("Drogued only", grid_dr)]):
        for ax, season in zip(axes[row], SEASONS):
            season_cell_map(ax, grid[grid["monsoon"] == season], "speed", cmap, norm,
                            f"{label}: {season}", arrows=True)
    add_land_legend(axes[0, 0], [INSUFFICIENT_PATCH])
    add_colorbar(fig, axes, cmap, norm, "Mean current speed (m/s)")
    fig.suptitle("Sensitivity of mean circulation to drogue status", fontsize=15)
    save_fig(fig, "07_circulation_all_vs_drogued.png")

    # Uncertainty: bootstrap standard error of the speed, drogued only.
    # Colour scale capped at the 95th percentile so a few extreme cells do not
    # wash out the rest; dots mark high-confidence cells.
    se_max = grid_dr["se_speed"].quantile(0.95)
    se_norm, se_cmap = Normalize(0, se_max), plt.get_cmap("Oranges")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
    for ax, season in zip(axes, SEASONS):
        season_cell_map(ax, grid_dr[grid_dr["monsoon"] == season], "se_speed", se_cmap, se_norm,
                        f"{season} monsoon")
        g = grid_dr[(grid_dr["monsoon"] == season) & (grid_dr["confidence"] == "high")]
        ax.scatter(g["lon"], g["lat"], marker=".", color="black", s=20, zorder=3)
    add_land_legend(axes[0], [INSUFFICIENT_PATCH, plt.Line2D([], [], marker=".", color="black",
                              linestyle="", label=f"High confidence (>={HIGH_CONF_DRIFTERS} drifters)")])
    add_colorbar(fig, axes, se_cmap, se_norm, "Bootstrap standard error of speed (m/s)")
    fig.suptitle("Uncertainty of the drogued-only mean current", fontsize=15)
    save_fig(fig, "08_uncertainty.png")


def reversal_analysis(per_dr, per_all, grid_dr, grid_all):
    """Headline result: the percentage of cells whose current reverses (> 90°)
    and strongly reverses (> 135°) between monsoons, with 95% confidence
    intervals, for the whole Bay and the EICC band, using both drogued-only and
    all data. Also draws the reversal map and writes its caption."""
    section(f"MONSOON REVERSAL (SW vs NE direction differs by > 90 deg, {N_BOOT} drifter bootstrap)")
    rows = []
    paired = {}
    for dataset, per, grid in [("drogued only", per_dr, grid_dr), ("all data", per_all, grid_all)]:
        p = paired_cells(grid)
        paired[dataset] = p
        for region, eicc_only in [("Whole Bay of Bengal", False),
                                  (f"Approx. EICC band {EICC_LON[0]:.0f}-{EICC_LON[1]:.0f}E", True)]:
            r = p[p["in_eicc_band"]] if eicc_only else p
            (lo, med, hi), (lo135, med135, hi135) = bootstrap_reversal(per, eicc_only=eicc_only)
            rows.append({"dataset": dataset, "region": region, "cells_both_seasons": len(r),
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

    cells = paired["drogued only"]
    print("\nDrogued-only cells by direction-change category:")
    print(pd.crosstab(cells["category"], cells["in_eicc_band"].map({True: "EICC band", False: "rest of bay"}),
                      margins=True))
    save_table(cells, "reversal_cells_drogued.csv")

    # Southern edge of the region (5-6°N): how well sampled are these cells?
    south = cells[cells["lat0"] == LAT_RANGE[0]][["lon0", "lat0", "angle_change", "category",
                                                  "n_drifters_sw", "n_drifters_ne"]]
    print(f"\nSouthern-edge cells ({LAT_RANGE[0]:.0f}-{LAT_RANGE[0] + GRID:.0f}N), drogued only:")
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
    setup_map(ax, "SW-to-NE change in current direction (drogued only)")
    add_land_legend(ax, [Patch(**no_data_style, label="No reliable data in one or both seasons")])
    cbar = add_colorbar(fig, ax, cmap, norm, "Direction change SW to NE (>90° = reversed)")
    cbar.set_ticks([(lo + hi) / 2 for lo, hi in zip(ANGLE_BINS[:-1], ANGLE_BINS[1:])],
                   labels=[f"{lo}-{hi}°\n{name.split(' (')[0]}"
                           for lo, hi, name in zip(ANGLE_BINS[:-1], ANGLE_BINS[1:], ANGLE_LABELS)])
    cbar.ax.tick_params(length=0)
    cbar.ax.axhline(90, color="black", linewidth=2)
    save_fig(fig, "09_reversal_map.png", dpi=300)

    caption = reversal_caption(summary[summary["dataset"] == "drogued only"])
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


def drogue_sensitivity(grid_all, grid_dr):
    """Robustness check: how much do the mean currents change when undrogued
    drifters are included? Compares speed and direction cell by cell, and checks
    whether the differences are larger where few drifters are available."""
    section("DROGUE SENSITIVITY: all data vs drogued only, cells sufficient in both")
    cols = CELL + ["ve", "vn", "speed", "n_drifters"]
    m = (grid_all[grid_all["sufficient"]][cols + ["lon", "lat"]]
         .merge(grid_dr[grid_dr["sufficient"]][cols], on=CELL, suffixes=("_all", "_dr")))
    m["speed_diff"] = m["speed_all"] - m["speed_dr"]
    m["abs_speed_diff"] = m["speed_diff"].abs()
    m["pct_speed_diff"] = np.where(m["speed_all"] >= MIN_SPEED_FOR_PCT,
                                   m["abs_speed_diff"] / m["speed_all"] * 100, np.nan)
    m["angle_diff"] = angle_between(m["ve_all"], m["vn_all"], m["ve_dr"], m["vn_dr"])
    m["sample_group"] = pd.cut(m["n_drifters_dr"], [0, 4, 9, np.inf], labels=["3-4", "5-9", "10+"])

    print(m.groupby("monsoon")[["abs_speed_diff", "pct_speed_diff", "angle_diff"]].describe().T.round(3))
    for season in SEASONS:
        d = m[m["monsoon"] == season]
        shares = ", ".join(f">{a} deg: {(d['angle_diff'] > a).mean() * 100:.1f}%" for a in (15, 30, 45, 90))
        print(f"\n{season}: direction change {shares}")
        print(f"  corr(n drogued drifters, |speed diff|) = {d['n_drifters_dr'].corr(d['abs_speed_diff']):.3f}")
        print(f"  corr(n drogued drifters, direction diff) = {d['n_drifters_dr'].corr(d['angle_diff']):.3f}")
    print("\nBy number of drogued drifters in the cell:")
    print(m.groupby(["monsoon", "sample_group"], observed=True)[["abs_speed_diff", "angle_diff"]]
           .agg(["count", "median", "mean", "max"]).round(3))
    save_table(m, "drogue_sensitivity_cells.csv")

    # Differences against the number of drogued drifters in the cell
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), sharex=True, constrained_layout=True)
    for row, season in enumerate(SEASONS):
        d = m[m["monsoon"] == season]
        axes[row, 0].scatter(d["n_drifters_dr"], d["abs_speed_diff"], alpha=0.5)
        axes[row, 0].set(ylabel="|Speed difference| (m/s)", title=f"{season}: speed")
        axes[row, 1].scatter(d["n_drifters_dr"], d["angle_diff"], alpha=0.5, color="darkorange")
        axes[row, 1].set(ylabel="Direction difference (deg)", title=f"{season}: direction")
    for ax in axes[1]:
        ax.set_xlabel("Drogued drifters in cell")
    fig.suptitle("All data vs drogued only, against drogued sample size", fontsize=15)
    save_fig(fig, "10_drogue_sensitivity_scatter.png")

    # Maps of where the differences are (green = small, red = large)
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    panels = [("pct_speed_diff", Normalize(0, 50), "% speed difference"),
              ("angle_diff", Normalize(0, 90), "Direction difference (deg)")]
    for row, season in enumerate(SEASONS):
        d = m[m["monsoon"] == season].assign(sufficient=True)
        for col, (value, norm, label) in enumerate(panels):
            season_cell_map(axes[row, col], d, value, plt.get_cmap("RdYlGn_r"), norm, f"{season}: {label}")
    for col, (_, norm, label) in enumerate(panels):
        add_colorbar(fig, axes[:, col], plt.get_cmap("RdYlGn_r"), norm, label)
    add_land_legend(axes[0, 0])
    fig.suptitle("Where drogue filtering changes the mean current", fontsize=15)
    save_fig(fig, "11_drogue_sensitivity_map.png")


def sst_analysis(drogued):
    """Gridded mean sea surface temperature for each monsoon, built the same way
    as the currents (drifter-weighted, same mapping rules), after removing SST
    readings with a reported error above ERR_SST_MAX."""
    section("SEA SURFACE TEMPERATURE (drogued only)")
    err = drogued["err_sst"].dropna()
    print(err.describe(percentiles=[0.5, 0.9, 0.95, 0.99]).round(3))
    print(f"err_sst > {ERR_SST_MAX}: {(err > ERR_SST_MAX).mean() * 100:.2f}% of non-null observations dropped")

    sst_obs = drogued.dropna(subset=["sst", "err_sst"])
    sst_obs = sst_obs[sst_obs["err_sst"] <= ERR_SST_MAX]
    grid = cell_grid(per_drifter_means(sst_obs, ["sst"]), ["sst"])
    ok = grid[grid["sufficient"]]
    print(ok.groupby("monsoon")["sst"].describe().round(2))
    save_table(grid, "sst_grid_drogued.csv")

    norm, cmap = Normalize(ok["sst"].min(), ok["sst"].max()), plt.get_cmap("YlOrRd")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
    for ax, season in zip(axes, SEASONS):
        season_cell_map(ax, grid[grid["monsoon"] == season], "sst", cmap, norm, f"{season} monsoon")
    add_land_legend(axes[0], [INSUFFICIENT_PATCH])
    add_colorbar(fig, axes, cmap, norm, "Mean sea surface temperature (°C)")
    fig.suptitle(f"Drifter-weighted mean SST (drogued only, err_sst <= {ERR_SST_MAX} °C)", fontsize=15)
    save_fig(fig, "12_sst.png", dpi=300)
    return grid


# ============================================================
# SPATIAL GAUSSIAN KERNEL SMOOTHING — DATA3001
# ============================================================
# Supplementary analysis of drogued surface currents.
# Original gridded estimates and reversal statistics are retained.

KERNEL_BANDWIDTHS_KM = (
    25.0, 40.0, 50.0, 60.0, 75.0, 90.0,
    100.0, 125.0, 175.0, 250.0, 350.0
)
KERNEL_TEST_FRACTION = 0.20
KERNEL_RADIUS_MULTIPLIER = 3.0


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


def kernel_smoothing_analysis(per_dr, grid_dr):
    """Run spatial kernel smoothing, validation and visualisation."""
    section("SPATIAL KERNEL SMOOTHING — DROGUED ONLY")

    drifter_ids = np.sort(per_dr["drifter_id"].unique())

    if len(drifter_ids) < 10:
        print("Too few drifters for kernel validation.")
        return

    rng = np.random.default_rng(SEED)
    shuffled = rng.permutation(drifter_ids)

    n_test = max(
        1, round(len(shuffled) * KERNEL_TEST_FRACTION)
    )

    test_ids = shuffled[:n_test]
    train_ids = shuffled[n_test:]

    train = per_dr[
        per_dr["drifter_id"].isin(train_ids)
    ].copy()

    test = per_dr[
        per_dr["drifter_id"].isin(test_ids)
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
    smoothed = grid_dr.copy()

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
        "kernel_smoothed_grid_drogued.csv"
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
        "Original vs kernel-smoothed circulation (drogued only)",
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


def temporal_weighting_analysis(drogued_all):
    """Monthly circulation maps using smooth temporal weighting, with the
    weighting width chosen by cross-validation and tested against the fixed
    seasons. Uses drogued observations from all twelve months."""
    section("SMOOTH TEMPORAL WEIGHTING (drogued only, all months)")
    rec = monthly_drifter_means(drogued_all)
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
    save_table(clim, "monthly_climatology_drogued.csv")
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
                 f"({best}, drogued drifters only)", fontsize=16)
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
                 f"(drogued drifters)")
    ax.legend(loc="best", fontsize=9)
    save_fig(fig, "17_eicc_annual_cycle.png")

    return {"best": best, "improvement": improvement, "ci": ci, "rmse": in_seasons}


def run():
    """Run the full analysis from start to finish."""
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
    drogued = season_obs[season_obs["drogue"]]         # main analysis data set
    exclusions = pd.concat([exclusions, pd.DataFrame([
        {"step": "inter-monsoon months (analysis only)", "removed": len(obs) - len(season_obs),
         "remaining": len(season_obs)},
        {"step": "undrogued (main analysis only)", "removed": len(season_obs) - len(drogued),
         "remaining": len(drogued)},
    ])], ignore_index=True)
    print(exclusions.to_string(index=False))
    save_table(exclusions, "data_exclusions.csv")
    print(f"\nTime span: {obs['time'].min():%Y-%m-%d} to {obs['time'].max():%Y-%m-%d}")

    explore(ds, obs)

    # Per-drifter cell means: drogued only (main) and all drifters (robustness check)
    per_dr = per_drifter_means(drogued)
    per_all = per_drifter_means(season_obs)
    threshold_sensitivity(per_dr)

    section("GRIDDED CLIMATOLOGY")
    grid_dr = add_bootstrap_se(velocity_grid(per_dr), per_dr)
    grid_all = add_bootstrap_se(velocity_grid(per_all), per_all)
    for label, g in [("Drogued only", grid_dr), ("All data", grid_all)]:
        print(f"\n{label}: cells by confidence tier")
        print(pd.crosstab(g["monsoon"], g["confidence"]))
    save_table(grid_dr, "climatology_grid_drogued.csv")
    save_table(grid_all, "climatology_grid_all.csv")

    # Kernel smoothing extension
    kernel_smoothing_analysis(per_dr, grid_dr) 

    circulation_figures(grid_dr, grid_all)
    reversal = reversal_analysis(per_dr, per_all, grid_dr, grid_all)
    drogue_sensitivity(grid_all, grid_dr)
    sst_grid = sst_analysis(drogued)

    # Alternative to fixed seasons: monthly maps with smooth temporal weighting
    temporal = temporal_weighting_analysis(obs[obs["drogue"]])

    # Headline numbers, written to summary.txt
    kept = grid_dr[grid_dr["sufficient"]]
    lines = [
        "HEADLINE NUMBERS",
        f"Rule: drogued only, {GRID:g} deg cells, >= {MIN_DRIFTERS} drifters each with >= "
        f"{MIN_OBS_PER_DRIFTER} hourly obs, drifter-weighted mean, {N_BOOT} whole-drifter bootstrap",
        "",
        f"Observations after cleaning: {len(obs)} ({obs['drifter_id'].nunique()} drifters), "
        f"{obs['time'].min():%Y-%m} to {obs['time'].max():%Y-%m}",
        f"SW + NE observations: {len(season_obs)}; drogued: {len(drogued)} "
        f"({len(drogued) / len(season_obs) * 100:.1f}%)",
        "",
        "Mapped cells (drogued):",
    ]
    for season in SEASONS:
        k = kept[kept["monsoon"] == season]
        lines.append(f"  {season}: {len(k)} cells ({(k['confidence'] == 'high').sum()} high confidence), "
                     f"SST cells: {int(sst_grid[(sst_grid['monsoon'] == season)]['sufficient'].sum())}")
    lines += ["", "SW vs NE reversal (> 90 deg):"]
    for _, r in reversal.iterrows():
        lines.append(f"  [{r['dataset']}] {r['region']}: {r['reversed_pct']:.1f}% of "
                     f"{r['cells_both_seasons']} cells (bootstrap median {r['boot_median_pct']:.1f}%, "
                     f"95% CI {r['ci95_low_pct']:.1f}-{r['ci95_high_pct']:.1f}%); "
                     f"> 135 deg: {r['gt135_pct']:.1f}% (median {r['gt135_boot_median_pct']:.1f}%, "
                     f"95% CI {r['gt135_ci95_low_pct']:.1f}-{r['gt135_ci95_high_pct']:.1f}%)")
    lines += ["", "Smooth temporal weighting vs fixed seasons (leave-one-drifter-out, SW + NE months):",
              f"  Selected: {temporal['best']}; RMSE {temporal['rmse'][temporal['best']]:.4f} m/s vs "
              f"{temporal['rmse'][FIXED_SEASONS]:.4f} m/s for fixed seasons, reduction "
              f"{temporal['improvement']:+.1f}% (95% CI {temporal['ci'][0]:+.1f} to {temporal['ci'][1]:+.1f}%)"]
    text = "\n".join(lines)
    section("SUMMARY")
    print(text)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nAll outputs written to {OUT_DIR}")


def main():
    """Create the output folders and run the analysis, saving everything printed
    to outputs/report.txt."""
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(TAB_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "report.txt"), "w", encoding="utf-8") as log:
        sys.stdout = Tee(sys.__stdout__, log)
        try:
            run()
        finally:
            sys.stdout = sys.__stdout__


if __name__ == "__main__":
    main()
