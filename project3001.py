r"""
Bay of Bengal surface circulation: Southwest vs Northeast monsoon,
from NOAA GDP hourly drifters (clouddrift).

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

LON_RANGE = (78.0, 99.0)
LAT_RANGE = (5.0, 23.0)
GRID = 1.0                       # grid cell size in degrees

SEASONS = ["Southwest", "Northeast"]
SEASON_OF_MONTH = {6: "Southwest", 7: "Southwest", 8: "Southwest", 9: "Southwest",
                   12: "Northeast", 1: "Northeast", 2: "Northeast"}

MAX_SPEED = 3.0                  # m/s, faster observations are dropped as unrealistic
MIN_OBS_PER_DRIFTER = 12         # hourly obs a drifter needs in a cell to count there
MIN_DRIFTERS = 3                 # qualifying drifters a cell needs to be mapped
HIGH_CONF_DRIFTERS = 10          # >= this many drifters = high confidence, else low
N_BOOT = 500
SEED = 0
EICC_LON = (80.0, 85.0)          # approximate EICC band: a longitude band, not a coastline mask
ERR_SST_MAX = 0.5                # deg C, SST observations with larger error are dropped
MIN_SPEED_FOR_PCT = 0.05         # m/s, % speed differences are only computed above this

# Threshold sensitivity grid
SENS_MIN_DRIFTERS = [1, 3, 5, 10]
SENS_MIN_OBS = [1, 6, 12, 24, 48]

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "bob_subset.nc")   # delete if LON_RANGE / LAT_RANGE change
OUT_DIR = os.path.join(HERE, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TAB_DIR = os.path.join(OUT_DIR, "tables")

CELL = ["monsoon", "lon0", "lat0"]   # lon0/lat0 = south-west corner of a grid cell

plt.rcParams.update({"font.size": 12})
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


# ============================================================
# Output helpers
# ============================================================

class Tee:
    """Send print() output to the console and to report.txt at the same time."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for s in self.streams:
            s.write(text)

    def flush(self):
        for s in self.streams:
            s.flush()


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def save_fig(fig, name, dpi=200):
    path = os.path.join(FIG_DIR, name)
    if os.path.exists(path):
        os.remove(path)     # overwriting a OneDrive placeholder in place fails with Errno 22
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> figures/{name}")


def save_table(df, name, index=False):
    df.to_csv(os.path.join(TAB_DIR, name), index=index)
    print(f"  -> tables/{name}")


# ============================================================
# Land overlay (Natural Earth coastline)
# ============================================================
# Land is drawn as filled coastline polygons above the data cells (so coastal
# cells are partly covered, like any gridded ocean map) and below arrows and
# outlines. Needs cartopy; the Natural Earth 50m coastline (~2 MB) is
# downloaded once. If that fails, a warning is printed and maps have no land.

LAND_COLOR = (0.78, 0.71, 0.54, 0.35)   # see-through sand, so coastal cells show through;
                                        # distinct from the grey "insufficient data" hatching
COAST_COLOR = "#4d4d4d"
LAND_ZORDER = 1.5           # data cells 0 < land < arrows / outlines 3

_land_polys = None   # None = not loaded yet, [] = loading failed


def _polygons_of(geom):
    if geom.geom_type == "Polygon":
        yield geom
    elif hasattr(geom, "geoms"):
        for g in geom.geoms:
            yield from _polygons_of(g)


def _get_land(margin=1.0):
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
    if _get_land():
        ax.add_collection(PolyCollection(_get_land(), facecolors=LAND_COLOR, edgecolors=COAST_COLOR,
                                         linewidths=0.6, zorder=LAND_ZORDER), autolim=False)


EICC_LINE_STYLE = dict(color="black", linestyle="--", linewidth=1.2, zorder=4)


def add_land_legend(ax, extra_handles=(), loc="upper left", fontsize=9):
    handles = list(extra_handles)
    if _get_land():
        handles.append(Patch(facecolor=LAND_COLOR, edgecolor=COAST_COLOR, label="Land"))
    handles.append(plt.Line2D([], [], **EICC_LINE_STYLE,
                              label=f"Approx. EICC band ({EICC_LON[0]:.0f}-{EICC_LON[1]:.0f}°E)"))
    ax.legend(handles=handles, loc=loc, fontsize=fontsize, framealpha=0.95)


# ============================================================
# Map helpers
# ============================================================

INSUFFICIENT_STYLE = dict(facecolor="lightgrey", edgecolor="grey", hatch="//", alpha=0.6, linewidth=0)
INSUFFICIENT_PATCH = Patch(**INSUFFICIENT_STYLE,
                           label=f"Insufficient data (<{MIN_DRIFTERS} drifters with "
                                 f">={MIN_OBS_PER_DRIFTER} obs)")


def _cell_polys(g):
    return [[(x, y), (x + GRID, y), (x + GRID, y + GRID), (x, y + GRID)]
            for x, y in zip(g["lon0"], g["lat0"])]


def setup_map(ax, title, xlim=LON_RANGE):
    ax.set_xlim(*xlim)
    ax.set_ylim(*LAT_RANGE)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    for lon in EICC_LON:
        ax.axvline(lon, **EICC_LINE_STYLE)


def draw_cells(ax, g, values, cmap, norm):
    if len(g):
        ax.add_collection(PolyCollection(_cell_polys(g), array=np.asarray(values, dtype=float),
                                         cmap=cmap, norm=norm, edgecolors="none", zorder=0),
                          autolim=False)


def draw_insufficient(ax, g):
    if len(g):
        ax.add_collection(PolyCollection(
            _cell_polys(g), facecolors=INSUFFICIENT_STYLE["facecolor"],
            edgecolors=INSUFFICIENT_STYLE["edgecolor"], hatch="//",
            alpha=INSUFFICIENT_STYLE["alpha"], linewidths=0, zorder=0), autolim=False)


def season_cell_map(ax, g, value_col, cmap, norm, title, arrows=False):
    """Colour sufficient cells by value_col, hatch insufficient ones, optional direction arrows."""
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
    cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                        location="right", shrink=0.85, pad=0.03)
    cbar.set_label(label)
    return cbar


# ============================================================
# Data loading and cleaning
# ============================================================

def load_subset():
    if os.path.exists(CACHE_PATH):
        print(f"Loading cached subset: {CACHE_PATH}")
        return xr.open_dataset(CACHE_PATH)

    import clouddrift as cd
    from clouddrift.ragged import subset
    print("Computing subset from gdp1h() (this takes a few minutes)...")
    ds = subset(cd.datasets.gdp1h(), {"lat": LAT_RANGE, "lon": LON_RANGE}, row_dim_name="traj")
    ds.load()
    ds.to_netcdf(CACHE_PATH)
    print("Saved subset to disk.")
    return ds


def data_quality_table(ds):
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
    """One row per hourly observation, with its drifter id. In the ragged array,
    observations are stored contiguously by drifter, so repeating each id by its
    rowsize lines the ids up with the observations."""
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
    """Drop missing / unrealistic / duplicate rows and report what was removed."""
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

    n_cells_x = int(round((LON_RANGE[1] - LON_RANGE[0]) / GRID))
    n_cells_y = int(round((LAT_RANGE[1] - LAT_RANGE[0]) / GRID))
    obs["lon0"] = LON_RANGE[0] + GRID * np.clip(np.floor((obs["lon"] - LON_RANGE[0]) / GRID), 0, n_cells_x - 1)
    obs["lat0"] = LAT_RANGE[0] + GRID * np.clip(np.floor((obs["lat"] - LAT_RANGE[0]) / GRID), 0, n_cells_y - 1)
    return obs, pd.DataFrame(steps)


# ============================================================
# Gridding: drifter-weighted cell means
# ============================================================

def per_drifter_means(obs, cols=("ve", "vn")):
    """One mean per drifter, per cell, per monsoon."""
    named = {c: (c, "mean") for c in cols}
    return (obs.groupby(CELL + ["drifter_id"])
               .agg(**named, n_obs=(cols[0], "count"))
               .reset_index())


def cell_grid(per, cols, min_obs=MIN_OBS_PER_DRIFTER, min_drifters=MIN_DRIFTERS):
    """Average the per-drifter means across drifters, so each drifter counts once.
    Only drifters with >= min_obs observations in the cell count. Returns every
    sampled cell, with 'sufficient' = at least min_drifters qualifying drifters."""
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
    g = cell_grid(per, ["ve", "vn"], min_obs, min_drifters)
    g["speed"] = np.hypot(g["ve"], g["vn"])
    g["direction_deg"] = np.degrees(np.arctan2(g["vn"], g["ve"]))   # 0 = east, 90 = north
    return g


def add_bootstrap_se(grid, per, seed=SEED):
    """Standard error of each sufficient cell's mean, by resampling its drifters
    with replacement, plus the high / low confidence tier."""
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

def in_eicc(lon0):
    return (lon0 >= EICC_LON[0]) & (lon0 < EICC_LON[1])


def angle_between(u1, v1, u2, v2):
    """Smallest angle between two vectors, 0-180 degrees."""
    d = np.degrees(np.arctan2(v2, u2) - np.arctan2(v1, u1))
    return np.abs((d + 180) % 360 - 180)


ANGLE_BINS = [0, 45, 90, 135, 180]
ANGLE_LABELS = ["little change (<45)", "moderate (45-90)", "strong (90-135)", "near-opposite (>135)"]
ANGLE_COLORS = ["#2166ac", "#92c5de", "#f4a582", "#b2182b"]   # blues = not reversed, reds = reversed


def paired_cells(grid):
    """Cells sufficient in BOTH monsoons, with the SW-to-NE direction change.
    Reversed = the two mean current vectors differ by more than 90 degrees."""
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
    """Resample whole drifters with replacement (a drifter drawn twice counts twice
    in the means) and recompute the % of both-season cells whose direction changes
    by more than each threshold. A cell still needs MIN_DRIFTERS *distinct* drifters
    per season, as in the point estimate. Returns a (len(thresholds), 3) array of
    2.5 / 50 / 97.5 percentiles."""
    q = per[per["n_obs"] >= MIN_OBS_PER_DRIFTER]
    if eicc_only:
        q = q[in_eicc(q["lon0"])]
    ids, drifter_idx = np.unique(q["drifter_id"].to_numpy(), return_inverse=True)
    cells, cell_idx = np.unique(q[["lon0", "lat0"]].to_numpy(), axis=0, return_inverse=True)
    code = cell_idx.ravel() * 2 + (q["monsoon"].to_numpy() == "Northeast")   # (cell, season) slot
    n_slots = len(cells) * 2
    ve, vn = q["ve"].to_numpy(), q["vn"].to_numpy()

    rng = np.random.default_rng(seed)
    results = []
    for _ in range(N_BOOT):
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
# Pipeline
# ============================================================

def explore(ds, obs):
    """Coverage, drogue status and speed diagnostics on the cleaned data (all months)."""
    section("COVERAGE")
    coverage = (obs.groupby("monsoon")
                   .agg(observations=("time", "size"), unique_drifters=("drifter_id", "nunique"),
                        drogued_pct=("drogue", "mean"))
                   .reindex(SEASONS + ["Inter-monsoon"]))
    coverage["drogued_pct"] *= 100
    print(coverage.round(1))
    save_table(coverage, "coverage_by_monsoon.csv", index=True)

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

    # Speed by drogue status: if undrogued drifters run faster, that supports
    # treating their velocities as wind-contaminated
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

    # Raw observation density vs distinct drifters per cell (fine 50 x 50 grid).
    # Raw counts reflect sampling effort; one looping drifter can inflate them.
    bins = [np.linspace(*LON_RANGE, 51), np.linspace(*LAT_RANGE, 51)]
    counts, xe, ye = np.histogram2d(obs["lon"], obs["lat"], bins=bins)
    ix = np.clip(np.searchsorted(xe, obs["lon"], side="right") - 1, 0, 49)
    iy = np.clip(np.searchsorted(ye, obs["lat"], side="right") - 1, 0, 49)
    first_visit = ~pd.DataFrame({"d": obs["drifter_id"].to_numpy(), "x": ix, "y": iy}).duplicated().to_numpy()
    unique, _, _ = np.histogram2d(obs["lon"][first_visit], obs["lat"][first_visit], bins=bins)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), constrained_layout=True)
    for ax, grid, cmap_name, label, title in [
        (axes[0], counts, "YlOrBr", "Observations per cell", "Observation density"),
        (axes[1], unique, "YlGnBu", "Distinct drifters per cell", "Distinct drifter coverage"),
    ]:
        g = np.where(grid == 0, np.nan, grid)
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad("white")
        im = ax.pcolormesh(xe, ye, g.T, cmap=cmap, shading="auto")
        add_land(ax)
        fig.colorbar(im, ax=ax, label=label)
        setup_map(ax, title)
    add_land_legend(axes[0])
    save_fig(fig, "04_density_and_unique_drifters.png")

    # Trajectories, first fix of each drifter in red
    fig, ax = plt.subplots(figsize=(10, 8))
    for _, d in obs.groupby("drifter_id", sort=False):
        ax.plot(d["lon"], d["lat"], linewidth=0.4, alpha=0.15, color="blue")
        ax.scatter(d["lon"].iloc[0], d["lat"].iloc[0], s=12, color="red", alpha=0.8)
    add_land(ax)
    add_land_legend(ax)
    setup_map(ax, "Drifter trajectories and first fix in region")
    save_fig(fig, "05_trajectories.png")


def threshold_sensitivity(per):
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
    section("CIRCULATION MAPS")
    vmax = np.ceil(pd.concat([grid_dr.loc[grid_dr["sufficient"], "speed"],
                              grid_all.loc[grid_all["sufficient"], "speed"]]).max() * 10) / 10
    norm, cmap = Normalize(0, vmax), plt.get_cmap("YlGnBu")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7.5), constrained_layout=True)
    for ax, season in zip(axes, SEASONS):
        season_cell_map(ax, grid_dr[grid_dr["monsoon"] == season], "speed", cmap, norm,
                        f"{season} monsoon", arrows=True)
    add_land_legend(axes[0], [INSUFFICIENT_PATCH])
    add_colorbar(fig, axes, cmap, norm, "Mean current speed (m/s)")
    fig.suptitle("Drifter-weighted mean surface circulation (drogued drifters only)", fontsize=15)
    save_fig(fig, "06_circulation_drogued.png", dpi=300)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    for row, (label, grid) in enumerate([("All data", grid_all), ("Drogued only", grid_dr)]):
        for ax, season in zip(axes[row], SEASONS):
            season_cell_map(ax, grid[grid["monsoon"] == season], "speed", cmap, norm,
                            f"{label}: {season}", arrows=True)
    add_land_legend(axes[0, 0], [INSUFFICIENT_PATCH])
    add_colorbar(fig, axes, cmap, norm, "Mean current speed (m/s)")
    fig.suptitle("Sensitivity of mean circulation to drogue status", fontsize=15)
    save_fig(fig, "07_circulation_all_vs_drogued.png")

    # Uncertainty: bootstrap standard error of the speed, drogued only
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

    # Southern edge of the domain: how well sampled are the 5-6N cells?
    south = cells[cells["lat0"] == LAT_RANGE[0]][["lon0", "lat0", "angle_change", "category",
                                                  "n_drifters_sw", "n_drifters_ne"]]
    print(f"\nSouthern-edge cells ({LAT_RANGE[0]:.0f}-{LAT_RANGE[0] + GRID:.0f}N), drogued only:")
    print(south.round(0).to_string(index=False))
    save_table(south, "reversal_southern_edge.csv")

    # Map of the SW-to-NE direction change per cell, binned so the 90 deg threshold is the colour split.
    # The headline numbers go in reversal_caption.txt (poster text), not in the title.
    norm, cmap = BoundaryNorm(ANGLE_BINS, len(ANGLE_COLORS)), ListedColormap(ANGLE_COLORS)
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    no_data_style = dict(facecolor="lightgrey", edgecolor="grey", hatch="//", alpha=0.6, linewidth=0)
    ax.add_patch(plt.Rectangle((LON_RANGE[0], LAT_RANGE[0]), np.ptp(LON_RANGE), np.ptp(LAT_RANGE),
                               **no_data_style, zorder=-1))
    if _get_land():   # opaque white land under the data cells, so the hatching stops at the coast
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
    """Poster caption text. If directions were random, 50% of cells would change by
    > 90 deg and 25% by > 135 deg, so a CI that includes those levels is flagged."""
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


def run():
    section("LOAD")
    ds = load_subset()
    print(f"Observations: {ds.sizes['obs']}, drifters: {ds.sizes['traj']}")

    section("DATA QUALITY (raw subset)")
    quality = data_quality_table(ds)
    print(quality.round(3))
    save_table(quality, "data_quality.csv", index=True)

    section("CLEANING")
    obs, exclusions = clean_obs(build_obs_table(ds))
    season_obs = obs[obs["monsoon"].isin(SEASONS)]
    drogued = season_obs[season_obs["drogue"]]
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

    circulation_figures(grid_dr, grid_all)
    reversal = reversal_analysis(per_dr, per_all, grid_dr, grid_all)
    drogue_sensitivity(grid_all, grid_dr)
    sst_grid = sst_analysis(drogued)

    # Headline numbers
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
    text = "\n".join(lines)
    section("SUMMARY")
    print(text)
    with open(os.path.join(OUT_DIR, "summary.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(f"\nAll outputs written to {OUT_DIR}")


def main():
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
