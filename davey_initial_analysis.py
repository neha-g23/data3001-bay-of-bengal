import os
 
import clouddrift as cd
from clouddrift.ragged import subset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
 
pd.set_option("display.width", 200)          # stop pandas truncating wide tables
pd.set_option("display.max_columns", None)
 
# =============================================================================
# SETTINGS (edit here only)
# =============================================================================
LON_MIN, LON_MAX = 78.0, 99.0
LAT_MIN, LAT_MAX = 5.0, 23.0
GRID = 1.0                 # cell size in degrees
MIN_DRIFTERS = 5           # reliable cell: at least this many distinct drifters ...
MIN_OBS = 100              # ... and at least this many observations (hourly data)
MAX_SPEED = 3.0            # m/s, drop speeds above this as outliers
DROGUED_ONLY = True        # main analysis uses drogued obs only (see section D for comparison)
SHOW_PLOTS = False         # False = only save figures to OUT_DIR (no windows to close); True = also pop them up
DRIFTER_WEIGHTED_MAPS = False  # True = each drifter counts equally in a cell mean (see section C)
 
CACHE_PATH = "bob_subset.nc"   # delete this file if you change the box above
OUT_DIR = "analysis_outputs"
os.makedirs(OUT_DIR, exist_ok=True)
 
SEASON_OF_MONTH = {12: "Northeast monsoon (Dec-Feb)", 1: "Northeast monsoon (Dec-Feb)",
                   2: "Northeast monsoon (Dec-Feb)",
                   3: "Pre-monsoon (Mar-May)", 4: "Pre-monsoon (Mar-May)", 5: "Pre-monsoon (Mar-May)",
                   6: "Southwest monsoon (Jun-Sep)", 7: "Southwest monsoon (Jun-Sep)",
                   8: "Southwest monsoon (Jun-Sep)", 9: "Southwest monsoon (Jun-Sep)",
                   10: "Post-monsoon (Oct-Nov)", 11: "Post-monsoon (Oct-Nov)"}
SEASONS = ["Northeast monsoon (Dec-Feb)", "Pre-monsoon (Mar-May)",
           "Southwest monsoon (Jun-Sep)", "Post-monsoon (Oct-Nov)"]
 
 
def savefig(name):
    plt.savefig(os.path.join(OUT_DIR, name), dpi=150, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close("all")
 
 
# =============================================================================
# 1. LOAD
# =============================================================================
if os.path.exists(CACHE_PATH):
    print("Loading cached subset...")
    ds_bob = xr.open_dataset(CACHE_PATH)
else:
    print("Computing subset (this will take a few mins)...")
    ds = cd.datasets.gdp1h()
    ds_bob = subset(ds, {"lat": (LAT_MIN, LAT_MAX), "lon": (LON_MIN, LON_MAX)}, row_dim_name="traj")
    ds_bob.load()
    ds_bob.to_netcdf(CACHE_PATH)
    print("Saved subset to disk.")
 
print(f"Bay of Bengal observations: {ds_bob.sizes['obs']}")
print(f"Number of distinct drifters: {ds_bob.sizes['traj']}")
 
# =============================================================================
# 2. ONE MASTER TABLE
# =============================================================================
drifter_id = np.repeat(ds_bob["id"].values, ds_bob["rowsize"].values.astype(int))
assert len(drifter_id) == ds_bob.sizes["obs"], "rowsize does not match number of observations"
 
df = pd.DataFrame({
    "drifter": drifter_id,
    "time": pd.to_datetime(ds_bob["time"].values),
    "lat": ds_bob["lat"].values,
    "lon": ds_bob["lon"].values,
    "ve": ds_bob["ve"].values,
    "vn": ds_bob["vn"].values,
    "drogue_status": ds_bob["drogue_status"].values,
})
 
print("\nMISSING VALUES (%):")
print((df.isna().mean() * 100).round(2))
print("\nSUMMARY (check for fill values / unrealistic ranges):")
print(df[["lat", "lon", "ve", "vn"]].describe())
 
# =============================================================================
# 3. LIGHT QC
# =============================================================================
n0 = len(df)
df = df.dropna(subset=["time", "lat", "lon", "ve", "vn"])
n_nan = n0 - len(df)
speed = np.hypot(df.ve, df.vn)
n_fast = int((speed >= MAX_SPEED).sum())
df = df[speed < MAX_SPEED]
df = df.drop_duplicates(subset=["drifter", "time"])
print(f"\nQC: removed {n_nan:,} rows with missing values, {n_fast:,} with speed >= {MAX_SPEED} m/s; "
      f"{len(df):,} of {n0:,} observations remain")
 
# drogue status: treat missing as undrogued, and say how many that affected
n_missing_drogue = int(df["drogue_status"].isna().sum())
df["drogued"] = df["drogue_status"].fillna(False).astype(bool)
print(f"Drogue status missing for {n_missing_drogue:,} observations (treated as undrogued)")
 
# seasons, years, grid cell indices
df["month"] = df.time.dt.month
df["year"] = df.time.dt.year
df["season"] = df.month.map(SEASON_OF_MONTH)
 
nx = int(round((LON_MAX - LON_MIN) / GRID))
ny = int(round((LAT_MAX - LAT_MIN) / GRID))
df["ix"] = np.clip(((df.lon - LON_MIN) // GRID).astype(int), 0, nx - 1)
df["iy"] = np.clip(((df.lat - LAT_MIN) // GRID).astype(int), 0, ny - 1)
lon_edges = LON_MIN + GRID * np.arange(nx + 1)
lat_edges = LAT_MIN + GRID * np.arange(ny + 1)
 
 
def to_grid(cells, col):
    """Per-cell table -> 2D (lat x lon) array for plotting."""
    z = np.full((ny, nx), np.nan)
    z[cells.iy.values, cells.ix.values] = cells[col].values
    return z
 
 
# =============================================================================
# 4. A. DATA AVAILABILITY
# =============================================================================
print("\n" + "=" * 70 + "\nA. DATA AVAILABILITY\n" + "=" * 70)
print(f"Observations: {len(df):,} | drifters: {df.drifter.nunique():,} | "
      f"{df.time.min().date()} to {df.time.max().date()}")
 
# observations AND unique drifters per season (your last block, now with 4 seasons)
season_summary = (df.groupby("season")
                    .agg(observations=("ve", "size"),
                         unique_drifters=("drifter", "nunique"),
                         years_covered=("year", "nunique"))
                    .join(df[df.drogued].groupby("season")
                            .agg(obs_drogued=("ve", "size"),
                                 drifters_drogued=("drifter", "nunique")))
                    .reindex(SEASONS))
print("\nPer season:\n", season_summary)
 
monthly = df.month.value_counts().sort_index()
yearly = df.year.value_counts().sort_index()
 
fig, ax = plt.subplots(1, 2, figsize=(15, 4.5), constrained_layout=True)
monthly.plot(kind="bar", color="steelblue", ax=ax[0]); ax[0].set_title("Observations per month")
ax[0].set_xlabel("Month"); ax[0].set_ylabel("Observations"); ax[0].tick_params(axis="x", rotation=0)
yearly.plot(kind="bar", ax=ax[1]); ax[1].set_title("Observations per year")
ax[1].set_xlabel("Year"); ax[1].set_ylabel("Observations")
savefig("A1_temporal_coverage.png")
 
# drogue status by season
drogue_tab = pd.crosstab(df.season, df.drogued, normalize="index").reindex(SEASONS) * 100
drogue_tab = drogue_tab.rename(columns={False: "Non-drogued", True: "Drogued"})
print("\nDrogue status by season (% of observations):\n", drogue_tab.round(1))
ax = drogue_tab.plot(kind="bar", stacked=True, figsize=(8, 4.5))
ax.set_ylabel("% of observations"); ax.set_title("Drogue status by season")
plt.xticks(rotation=15); plt.tight_layout()
savefig("A2_drogue_by_season.png")
 
# trajectories (as in your version, but drogued/undrogued not distinguished here)
fig = plt.figure(figsize=(9, 8))
for _, g in df.sort_values(["drifter", "time"]).groupby("drifter"):
    plt.plot(g.lon, g.lat, lw=0.4, alpha=0.12, color="blue")
    plt.scatter(g.lon.iloc[0], g.lat.iloc[0], s=12, color="red", alpha=0.7)
plt.xlim(LON_MIN, LON_MAX); plt.ylim(LAT_MIN, LAT_MAX); plt.grid(alpha=0.2)
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.title("Drifter trajectories and first positions inside the region")
savefig("A3_trajectories.png")
 
# =============================================================================
# 5. B. SAMPLING BIAS: raw obs density vs unique drifter density
# =============================================================================
print("\n" + "=" * 70 + "\nB. SAMPLING BIAS\n" + "=" * 70)
main = df[df.drogued] if DROGUED_ONLY else df
print(f"Main analysis uses {'drogued-only' if DROGUED_ONLY else 'all'} data: "
      f"{len(main):,} observations, {main.drifter.nunique():,} drifters")
 
allcells = (main.groupby(["iy", "ix"])
                .agg(n_obs=("ve", "size"), n_drifters=("drifter", "nunique")).reset_index())
allcells["obs_per_drifter"] = allcells.n_obs / allcells.n_drifters
 
fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
for ax, (col, title, cmap) in zip(axes, [
        ("n_obs", "Raw observation density (obs per cell)", "YlOrBr"),
        ("n_drifters", "Unique drifter density (drifters per cell)", "magma"),
        ("obs_per_drifter", "Observations per drifter (high = a few drifters dominate)", "cividis")]):
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("lightblue")
    m = ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_invalid(to_grid(allcells, col)), cmap=cm)
    ax.set_title(title, fontsize=10); ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
    fig.colorbar(m, ax=ax, shrink=0.8)
savefig("B_density_raw_vs_unique.png")
 
# =============================================================================
# 6. C. GRIDDED SEASONAL STATISTICS + RELIABILITY MASK
# =============================================================================
print("\n" + "=" * 70 + "\nC. SEASONAL GRIDDED MEAN CURRENT + RELIABILITY\n" + "=" * 70)
 
 
def cell_stats(data):
    c = (data.groupby(["season", "iy", "ix"])
             .agg(n_obs=("ve", "size"), n_drifters=("drifter", "nunique"),
                  mean_ve=("ve", "mean"), mean_vn=("vn", "mean")).reset_index())
    # drifter-weighted mean: average each drifter's own mean first, then across drifters,
    # so one drifter that lingers in a cell cannot dominate it
    per_d = (data.groupby(["season", "iy", "ix", "drifter"])
                 .agg(ve=("ve", "mean"), vn=("vn", "mean")).reset_index())
    dw = (per_d.groupby(["season", "iy", "ix"])
               .agg(dw_ve=("ve", "mean"), dw_vn=("vn", "mean")).reset_index())
    c = c.merge(dw, on=["season", "iy", "ix"])
    c["speed"] = np.hypot(c.mean_ve, c.mean_vn)
    c["dw_speed"] = np.hypot(c.dw_ve, c.dw_vn)
    c["reliable"] = (c.n_drifters >= MIN_DRIFTERS) & (c.n_obs >= MIN_OBS)
    return c
 
 
cells = cell_stats(main)
 
rows = []
for s in SEASONS:
    c = cells[cells.season == s]
    rows.append({"season": s, "cells_sampled": len(c), "cells_reliable": int(c.reliable.sum()),
                 "pct_reliable": round(100 * c.reliable.mean(), 1) if len(c) else np.nan,
                 "median_drifters_per_cell": c.n_drifters.median(),
                 "median_obs_per_cell": c.n_obs.median()})
reliability_table = pd.DataFrame(rows).set_index("season")
print(f"\nGrid {GRID} deg = {nx * ny} cells; reliable = >= {MIN_DRIFTERS} drifters and >= {MIN_OBS} obs")
print(reliability_table)
 
# does it matter whether each observation or each drifter is weighted equally?
_r = cells[cells.reliable]
if len(_r):
    _d = np.hypot(_r.mean_ve - _r.dw_ve, _r.mean_vn - _r.dw_vn)
    print(f"\nObs-weighted vs drifter-weighted mean (reliable cells): median vector difference "
          f"{_d.median():.3f} m/s, 90th percentile {_d.quantile(0.9):.3f} m/s, "
          f"median speed {_r.speed.median():.3f} m/s")
 
# maps: coloured + arrows = reliable; grey = sampled but unreliable
reliable_cells = cells[cells.reliable]
VE, VN, SP = (("dw_ve", "dw_vn", "dw_speed") if DRIFTER_WEIGHTED_MAPS
              else ("mean_ve", "mean_vn", "speed"))
if reliable_cells.empty:
    print("No reliable cells: lower MIN_OBS / MIN_DRIFTERS or increase GRID.")
else:
    vmax = reliable_cells[SP].quantile(0.98)
    LON, LAT = np.meshgrid(lon_edges[:-1] + GRID / 2, lat_edges[:-1] + GRID / 2)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11), constrained_layout=True)
    for ax, s in zip(axes.ravel(), SEASONS):
        c = cells[cells.season == s]
        r = c[c.reliable]
        ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_invalid(to_grid(c[~c.reliable].assign(g=1.0), "g")),
                      cmap="Greys", vmin=0, vmax=3)
        m = ax.pcolormesh(lon_edges, lat_edges, np.ma.masked_invalid(to_grid(r, SP)),
                          cmap="YlGnBu", vmin=0, vmax=vmax)
        ve, vn = to_grid(r, VE), to_grid(r, VN)
        ok = ~np.isnan(ve)
        sp = np.maximum(np.hypot(ve[ok], vn[ok]), 1e-9)       # avoid divide-by-zero
        ax.quiver(LON[ok], LAT[ok], ve[ok] / sp, vn[ok] / sp,       # unit arrows, speed = colour
                  angles="xy", scale_units="xy", scale=1.2, width=0.003, color="k")
        ax.set_title(f"{s}  ({int(r.shape[0])} reliable cells)")
        ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX); ax.grid(alpha=0.2)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        fig.colorbar(m, ax=ax, shrink=0.8, label="Mean speed (m/s)")
    fig.suptitle("Mean surface current by season: arrows = direction, colour = speed; "
                 "grey = sampled but unreliable", fontsize=13)
    savefig("C_seasonal_mean_current.png")
 
# =============================================================================
# 7. D. EVIDENCE FOR THE PLAN: drogued-only vs all drifters
# =============================================================================
print("\n" + "=" * 70 + "\nD. DROGUED-ONLY vs ALL DRIFTERS\n" + "=" * 70)
cmp_rows = []
for label, data in [("drogued only", df[df.drogued]), ("all drifters", df)]:
    cc = cell_stats(data)
    for s in SEASONS:
        c = cc[cc.season == s]
        d = data[data.season == s]
        cmp_rows.append({"data": label, "season": s, "observations": len(d),
                         "drifters": d.drifter.nunique(),
                         "cells_reliable": int(c.reliable.sum())})
comparison = pd.DataFrame(cmp_rows).pivot(index="season", columns="data",
                                          values=["observations", "drifters", "cells_reliable"])
print(comparison.reindex(SEASONS))
 
# =============================================================================
# 8. SAVE TABLES
# =============================================================================
season_summary.to_csv(os.path.join(OUT_DIR, "season_summary.csv"))
reliability_table.to_csv(os.path.join(OUT_DIR, "reliability_by_season.csv"))
comparison.reindex(SEASONS).to_csv(os.path.join(OUT_DIR, "drogued_vs_all.csv"))
cells.to_csv(os.path.join(OUT_DIR, "cell_stats_by_season.csv"), index=False)
print(f"\nDone. Figures and tables saved in {OUT_DIR}/")
