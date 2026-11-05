# ============================================================
# Kernel-smoothed seasonal circulation, Bay of Bengal (standalone)
# ============================================================
# Extracted from project3001.py so it runs on its own:
#   - data loading (section 1)
#   - drifter-weighted per-drifter means (section 13.1-13.4)
#   - kernel smoothing, bandwidth cross-validation, maps and
#     comparison with the gridded estimate (section 14)
#
# Needs bob_subset.nc in the same folder (or it rebuilds it from the
# full GDP hourly dataset, which takes a while).
# Install: pip install -r requirements.txt
# ============================================================

import os
import clouddrift as cd
from clouddrift.ragged import subset
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize, ListedColormap
import pandas as pd
import numpy as np

# ============================================================
# 1. Load Bay of Bengal (BoB) subset, caching it to disk
# ============================================================
# gdp1h() streams the full global drifter dataset, so the BoB subset is
# computed once and saved locally; later runs just load the cached file.
# Delete bob_subset.nc if the region bounds below ever change.

cache_path = "bob_subset.nc"

if os.path.exists(cache_path):
    print("Loading cached subset...")
    import xarray as xr
    ds_bob = xr.open_dataset(cache_path)
else:
    print("Computing subset (this will take a few mins)...")
    ds = cd.datasets.gdp1h()
    ds_bob = subset(
        ds,
        {"lat": (5.0, 23.0), "lon": (78.0, 99.0)},
        row_dim_name="traj"
    )
    ds_bob.load()  # pull fully into memory before saving to netCDF
    ds_bob.to_netcdf(cache_path)
    print("Saved subset to disk.")

print(f"Bay of Bengal observations: {ds_bob.sizes['obs']}")

n_drifters = ds_bob.sizes['traj']
print(f"Number of distinct drifters: {n_drifters}")



# ============================================================
# 13. Drifter-weighted seasonal circulation map: all data vs drogued-only
# ============================================================
# Corrects the naive map in section 10: averages within each drifter per
# cell first, then across drifters, so one heavily-sampled drifter can't
# dominate a cell's mean current. Also compares the full dataset against a
# drogued-only subset, since post-drogue-loss velocities are wind-driven
# rather than current-driven (see section 8's speed diagnostic). Cells
# below a minimum drifter count are shaded as "insufficient data" instead
# of being silently dropped.

print("\nDRIFTER-WEIGHTED MONSOON CIRCULATION: ALL DATA VS DROGUED-ONLY\n")

# --------------------------------------------------
# 13.1 Attach a drifter ID to every observation
# --------------------------------------------------

drifter_ids = np.repeat(
    ds_bob["id"].values,
    ds_bob["rowsize"].values.astype(int)
)

currents = pd.DataFrame({
    "time": ds_bob["time"].values,
    "lat": ds_bob["lat"].values,
    "lon": ds_bob["lon"].values,
    "ve": ds_bob["ve"].values,
    "vn": ds_bob["vn"].values,
    "drogue_status": ds_bob["drogue_status"].values,
    "drifter_id": drifter_ids
})

currents = currents.dropna(
    subset=["time", "lat", "lon", "ve", "vn"]
)

# --------------------------------------------------
# 13.2 Label monsoon period
# --------------------------------------------------

currents["month"] = pd.to_datetime(currents["time"]).dt.month

def get_monsoon(month):
    if month in [6, 7, 8, 9]:
        return "Southwest"
    elif month in [12, 1, 2]:
        return "Northeast"
    else:
        return "Inter-monsoon"

currents["monsoon"] = currents["month"].apply(get_monsoon)

# Only SW and NE for the main comparison
currents = currents[
    currents["monsoon"].isin(["Southwest", "Northeast"])
].copy()

# --------------------------------------------------
# 13.3 Create 1° x 1° grid
# --------------------------------------------------

grid_size = 1.0

lat_bins = np.arange(5, 23 + grid_size, grid_size)
lon_bins = np.arange(78, 99 + grid_size, grid_size)

currents["lat_bin"] = pd.cut(
    currents["lat"],
    bins=lat_bins,
    include_lowest=True
)

currents["lon_bin"] = pd.cut(
    currents["lon"],
    bins=lon_bins,
    include_lowest=True
)

# Drogued-only subset, same bins as above
currents_drogued = currents[currents["drogue_status"]].copy()

# --------------------------------------------------
# 13.4 Drifter-weighted grid: average within each drifter first, then
#      across drifters. Reused for both the all-data and drogued-only
#      subsets so the two comparisons follow an identical procedure.
# --------------------------------------------------

MIN_OBS_PER_DRIFTER = 12

def build_drifter_weighted_grid(currents_subset):
    drifter_cell = (
        currents_subset
        .groupby(
            ["monsoon", "lat_bin", "lon_bin", "drifter_id"],
            observed=True
        )
        .agg(
            drifter_ve=("ve", "mean"),
            drifter_vn=("vn", "mean"),
            drifter_obs=("ve", "count"),
            drifter_lat=("lat", "mean"),
            drifter_lon=("lon", "mean")
        )
        .reset_index()
    )

    drifter_cell = drifter_cell[drifter_cell["drifter_obs"] >= MIN_OBS_PER_DRIFTER]

    cell_grid = (
        drifter_cell
        .groupby(
            ["monsoon", "lat_bin", "lon_bin"],
            observed=True
        )
        .agg(
            mean_ve=("drifter_ve", "mean"),
            mean_vn=("drifter_vn", "mean"),
            n_drifters=("drifter_id", "nunique"),
            n_obs=("drifter_obs", "sum")
        )
        .reset_index()
    )

    all_cells = currents_subset[["monsoon", "lat_bin", "lon_bin"]].drop_duplicates()
    cell_grid = all_cells.merge(
        cell_grid, on=["monsoon", "lat_bin", "lon_bin"], how="left"
    )
    cell_grid[["n_drifters", "n_obs"]] = cell_grid[["n_drifters", "n_obs"]].fillna(0).astype(int)

    cell_grid["lat_center"] = cell_grid["lat_bin"].apply(lambda x: x.mid).astype(float)
    cell_grid["lon_center"] = cell_grid["lon_bin"].apply(lambda x: x.mid).astype(float)

    return cell_grid, drifter_cell  # now returns both

grid_all, drifter_cell_all = build_drifter_weighted_grid(currents)
grid_drogued, drifter_cell_drogued = build_drifter_weighted_grid(currents_drogued)

# --------------------------------------------------

# Minimum drifters per 1° cell for the gridded comparison in 14.7
# (set in section 13.5 of project3001.py)
min_drifters = 3

# ============================================================
# 14. Kernel-smoothed (distance-weighted) seasonal circulation
# ============================================================
# Section 13 averages only the data inside each 1° box, so neighbouring
# boxes can point in very different directions just because of noise.
# Here each location's current is a weighted average of ALL nearby
# per-drifter means: close data get a large weight, far data a small one
# (Gaussian kernel on great-circle distance). The kernel width (bandwidth)
# is chosen by cross-validation that holds out whole drifters, because
# observations from one drifter are not independent.
#
# Input: drifter_cell_drogued from 13.4 (one row per drifter per cell,
# already averaged within the drifter, so the drifter weighting is kept).

from sklearn.model_selection import GroupKFold

EARTH_RADIUS_KM = 6371.0

# --------------------------------------------------
# 14.1 Distance and kernel-smoothing helpers
# --------------------------------------------------

def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance matrix (km): rows = points 1, columns = points 2."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    dlat = lat2[None, :] - lat1[:, None]
    dlon = lon2[None, :] - lon1[:, None]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(lat1)[:, None] * np.cos(lat2)[None, :] * np.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def kernel_smooth(target_lat, target_lon, pts, bandwidth_km, cutoff=3.0):
    """Gaussian-weighted mean of ve and vn at each target location.

    pts needs columns drifter_lat, drifter_lon, drifter_ve, drifter_vn,
    drifter_id. Returns ve, vn, and the number of distinct drifters within
    one bandwidth (used to mask poorly supported locations).
    """
    d = haversine_km(target_lat, target_lon,
                     pts["drifter_lat"].values, pts["drifter_lon"].values)
    w = np.exp(-0.5 * (d / bandwidth_km) ** 2)
    w[d > cutoff * bandwidth_km] = 0.0          # ignore data far away

    w_sum = w.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ve = w @ pts["drifter_ve"].values / w_sum
        vn = w @ pts["drifter_vn"].values / w_sum

    # Distinct drifters within one bandwidth of each target
    codes, _ = pd.factorize(pts["drifter_id"])
    onehot = np.zeros((len(pts), codes.max() + 1))
    onehot[np.arange(len(pts)), codes] = 1.0
    near = (d <= bandwidth_km).astype(float)
    n_drifters_near = ((near @ onehot) > 0).sum(axis=1)

    return ve, vn, n_drifters_near


# --------------------------------------------------
# 14.2 Points to smooth: per-drifter means at their own mean positions
# --------------------------------------------------
# Needs drifter_lat / drifter_lon, which 13.4 does not keep yet:
# add these two lines to the .agg(...) in build_drifter_weighted_grid
#     drifter_lat=("lat", "mean"),
#     drifter_lon=("lon", "mean"),

points = drifter_cell_drogued.dropna(
    subset=["drifter_lat", "drifter_lon", "drifter_ve", "drifter_vn"]
).copy()

# --------------------------------------------------
# 14.3 Choose the bandwidth: leave-drifters-out cross-validation
# --------------------------------------------------
# GroupKFold puts each drifter entirely in either the training or the
# test fold. Every bandwidth is scored on the same test points (those it
# can predict at the smallest bandwidth), so the comparison is fair.

BANDWIDTHS_KM = [25, 35, 50, 60, 75, 85, 100, 125, 150]
N_FOLDS = 10


def cv_bandwidths(season_points, bandwidths, n_folds=N_FOLDS):
    preds = {h: np.full((len(season_points), 2), np.nan) for h in bandwidths}
    gkf = GroupKFold(n_splits=n_folds)
    for train_idx, test_idx in gkf.split(season_points,
                                         groups=season_points["drifter_id"]):
        train = season_points.iloc[train_idx]
        test = season_points.iloc[test_idx]
        for h in bandwidths:
            ve, vn, _ = kernel_smooth(test["drifter_lat"].values,
                                      test["drifter_lon"].values, train, h)
            preds[h][test_idx] = np.column_stack([ve, vn])

    common = np.all([np.isfinite(preds[h]).all(axis=1) for h in bandwidths], axis=0)
    obs = season_points[["drifter_ve", "drifter_vn"]].values

    rows = []
    for h in bandwidths:
        sq_err = ((preds[h] - obs) ** 2).sum(axis=1)[common]   # vector error²
        # Standard error at the drifter level (errors within a drifter are correlated)
        per_drifter = (pd.Series(sq_err, index=season_points["drifter_id"].values[common])
                       .groupby(level=0).mean())
        mse = sq_err.mean()
        se_mse = per_drifter.std(ddof=1) / np.sqrt(len(per_drifter))
        rows.append({"bandwidth_km": h,
                     "rmse": np.sqrt(mse),
                     "rmse_se": se_mse / (2 * np.sqrt(mse)),   # delta method
                     "n_test_points": int(common.sum())})
    return pd.DataFrame(rows)


cv_results = {}
for season in ["Southwest", "Northeast"]:
    sp = points[points["monsoon"] == season].reset_index(drop=True)
    cv_results[season] = cv_bandwidths(sp, BANDWIDTHS_KM)
    print(f"\nCROSS-VALIDATION — {season}")
    print(cv_results[season].round(4).to_string(index=False))

# One-standard-error rule, applied to each season separately: the LARGEST
# bandwidth whose RMSE is within one SE of that season's best. This picks
# the smoothest map that is not measurably worse at prediction. The final
# bandwidth is the smaller of the two seasons' choices, so it is within
# one SE of the best in both monsoons.
one_se_choice = {}
for season, r in cv_results.items():
    best = r.loc[r["rmse"].idxmin()]
    one_se_choice[season] = r.loc[
        r["rmse"] <= best["rmse"] + best["rmse_se"], "bandwidth_km"
    ].max()
    print(f"\n{season}: lowest-RMSE bandwidth {best['bandwidth_km']:.0f} km; "
          f"1-SE rule: {one_se_choice[season]:.0f} km")

BANDWIDTH_KM = min(one_se_choice.values())
print(f"Chosen bandwidth (smaller of the two): {BANDWIDTH_KM:.0f} km")

# --------------------------------------------------
# 14.4 Bandwidth-selection figure
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
for season, colour in [("Southwest", "#d95f02"), ("Northeast", "#1b9e77")]:
    r = cv_results[season]
    ax.errorbar(r["bandwidth_km"], r["rmse"], yerr=r["rmse_se"],
                marker="o", capsize=3, color=colour, label=season)
ax.axvline(BANDWIDTH_KM, color="grey", ls="--", lw=1)
ax.text(BANDWIDTH_KM, ax.get_ylim()[1], f" chosen: {BANDWIDTH_KM:.0f} km",
        va="top", fontsize=9, color="grey")
ax.set_xlabel("Kernel bandwidth (km)")
ax.set_ylabel("Held-out vector RMSE (m/s)")
ax.set_title("Bandwidth selection (drifters held out)")
ax.legend(frameon=False)

# --------------------------------------------------
# 14.5 Smoothed field on a 0.5° target grid
# --------------------------------------------------

TARGET_RES = 0.5
MIN_DRIFTERS_NEAR = 3       # same spirit as min_drifters in section 13

t_lat_1d = np.arange(5 + TARGET_RES / 2, 23, TARGET_RES)
t_lon_1d = np.arange(78 + TARGET_RES / 2, 99, TARGET_RES)
t_lon_2d, t_lat_2d = np.meshgrid(t_lon_1d, t_lat_1d)

# Land mask (pip install global-land-mask)
from global_land_mask import globe
is_land = globe.is_land(t_lat_2d, t_lon_2d)

smoothed = {}
for season in ["Southwest", "Northeast"]:
    sp = points[points["monsoon"] == season]
    ve, vn, n_near = kernel_smooth(t_lat_2d.ravel(), t_lon_2d.ravel(),
                                   sp, BANDWIDTH_KM)
    ve, vn, n_near = (a.reshape(t_lat_2d.shape) for a in (ve, vn, n_near))
    low_support = (n_near < MIN_DRIFTERS_NEAR) & ~is_land
    ve[is_land | low_support] = np.nan
    vn[is_land | low_support] = np.nan
    smoothed[season] = {"ve": ve, "vn": vn,
                        "speed": np.hypot(ve, vn),
                        "low_support": low_support}

# --------------------------------------------------
# 14.6 Plot: smoothed circulation, SW vs NE (pip install cartopy)
# --------------------------------------------------

import cartopy.crs as ccrs
import cartopy.feature as cfeature

speed_max = np.nanpercentile(
    np.concatenate([smoothed[s]["speed"].ravel() for s in smoothed]), 98)
norm = Normalize(vmin=0, vmax=speed_max)
cmap = plt.get_cmap("viridis")
ARROW_STEP = 2              # draw every 2nd arrow (every 1°) to reduce clutter

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True,
                         subplot_kw={"projection": ccrs.PlateCarree()})

for ax, season, letter in zip(axes, ["Southwest", "Northeast"], ["a", "b"]):
    f = smoothed[season]
    ax.set_extent([78, 99, 5, 23], crs=ccrs.PlateCarree())

    # Low-support ocean: light grey, so "not enough data" is visible
    ax.pcolormesh(t_lon_2d, t_lat_2d,
                  np.where(f["low_support"], 1.0, np.nan),
                  cmap=ListedColormap(["#e6e6e6"]), shading="auto",
                  transform=ccrs.PlateCarree(), zorder=0)
    mesh = ax.pcolormesh(t_lon_2d, t_lat_2d, f["speed"], cmap=cmap, norm=norm,
                         shading="auto", transform=ccrs.PlateCarree(), zorder=1)

    s = (slice(None, None, ARROW_STEP), slice(None, None, ARROW_STEP))
    q = ax.quiver(t_lon_2d[s], t_lat_2d[s], f["ve"][s], f["vn"][s],
                  color="white", edgecolor="black", linewidth=0.3,
                  scale=8, width=0.0035, transform=ccrs.PlateCarree(), zorder=3)

    ax.add_feature(cfeature.LAND, facecolor="#bdbdbd", zorder=2)
    ax.coastlines("50m", linewidth=0.6, zorder=2)
    ax.plot([80, 85, 85, 80, 80], [5, 5, 23, 23, 5], color="red", lw=1,
            ls="--", transform=ccrs.PlateCarree(), zorder=4)   # EICC band
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.top_labels = gl.right_labels = False
    ax.set_title(f"({letter}) {season} monsoon", fontsize=13)

axes[1].quiverkey(q, 0.86, -0.08, 0.5, "0.5 m/s", labelpos="E",
                  coordinates="axes")

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes,
                    location="right", shrink=0.8, pad=0.02)
cbar.set_label("Current speed (m/s)")

fig.suptitle(f"Kernel-smoothed, drifter-weighted surface circulation "
             f"(drogued only, {BANDWIDTH_KM:.0f} km bandwidth)", fontsize=15)
# Caption idea: "Arrows show current direction and speed; grey ocean has
# fewer than 3 drifters within one bandwidth. Dashed box: approximate EICC band."

# --------------------------------------------------
# 14.7 Does smoothing predict better than the section 13 grid?
# --------------------------------------------------
# Both methods are scored on the same held-out drifters, in the same
# GroupKFold folds as 14.3. Each one is built from the training drifters
# only, with the same "at least 3 drifters" rule as its map, and scored
# only on test points that both methods can predict.
# Caveat: BANDWIDTH_KM was chosen using these same drifters, so the
# smoother's score is slightly optimistic; the per-bandwidth table below
# shows how much the result depends on that choice.

def compare_with_grid(season_points, bandwidth_km, n_folds=N_FOLDS):
    n = len(season_points)
    grid_pred = np.full((n, 2), np.nan)
    kernel_pred = np.full((n, 2), np.nan)
    cell_keys = ["lat_bin", "lon_bin"]

    gkf = GroupKFold(n_splits=n_folds)
    for train_idx, test_idx in gkf.split(season_points,
                                         groups=season_points["drifter_id"]):
        train = season_points.iloc[train_idx]
        test = season_points.iloc[test_idx]

        # Section 13 method: mean of the training drifters in the same cell
        train_cells = (train.groupby(cell_keys, observed=True)
                       .agg(ve=("drifter_ve", "mean"), vn=("drifter_vn", "mean"),
                            n=("drifter_id", "nunique"))
                       .reset_index())
        train_cells = train_cells[train_cells["n"] >= min_drifters]
        matched = test[cell_keys].merge(train_cells, on=cell_keys, how="left")
        grid_pred[test_idx] = matched[["ve", "vn"]].values

        # Section 14 method: kernel average, masked as on the map
        ve, vn, n_near = kernel_smooth(test["drifter_lat"].values,
                                       test["drifter_lon"].values,
                                       train, bandwidth_km)
        supported = n_near >= MIN_DRIFTERS_NEAR
        kernel_pred[test_idx[supported]] = np.column_stack([ve, vn])[supported]

    both = np.isfinite(grid_pred).all(axis=1) & np.isfinite(kernel_pred).all(axis=1)
    obs_uv = season_points[["drifter_ve", "drifter_vn"]].values
    grid_sq = ((grid_pred - obs_uv) ** 2).sum(axis=1)[both]
    kernel_sq = ((kernel_pred - obs_uv) ** 2).sum(axis=1)[both]
    ids_both = season_points["drifter_id"].values[both]

    # Bootstrap across drifters (fixed seed) for the RMSE difference
    errors = pd.DataFrame({"id": ids_both, "grid": grid_sq, "kernel": kernel_sq})
    totals = errors.groupby("id")[["grid", "kernel"]].sum()
    counts = errors.groupby("id").size()
    boot_rng = np.random.default_rng(3001)
    boot_diff = []
    for _ in range(2000):
        pick = boot_rng.integers(0, len(totals), len(totals))
        n_pts = counts.values[pick].sum()
        boot_diff.append(np.sqrt(totals["grid"].values[pick].sum() / n_pts)
                         - np.sqrt(totals["kernel"].values[pick].sum() / n_pts))
    ci_low, ci_high = np.percentile(boot_diff, [2.5, 97.5])

    return {
        "bandwidth_km": bandwidth_km,
        "n_test_points": int(both.sum()),
        "n_test_drifters": int(len(totals)),
        "grid_rmse": np.sqrt(grid_sq.mean()),
        "kernel_rmse": np.sqrt(kernel_sq.mean()),
        "improvement": np.sqrt(grid_sq.mean()) - np.sqrt(kernel_sq.mean()),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "grid_predictable": int(np.isfinite(grid_pred).all(axis=1).sum()),
        "kernel_predictable": int(np.isfinite(kernel_pred).all(axis=1).sum()),
        "all_points": n,
    }


for season in ["Southwest", "Northeast"]:
    sp = points[points["monsoon"] == season].reset_index(drop=True)
    comparison = pd.DataFrame(
        [compare_with_grid(sp, h) for h in BANDWIDTHS_KM]
    )
    print(f"\nHELD-OUT RMSE, GRID VS SMOOTHED — {season} "
          f"(chosen bandwidth: {BANDWIDTH_KM:.0f} km)")
    print(comparison.round(4).to_string(index=False))

# --------------------------------------------------
# 14.8 Kernel-smoothed unique-drifter density (coverage behind 14.6)
# --------------------------------------------------
# Smoothed version of the section 12 unique-drifter map, using the same
# drogued per-drifter points and the same Gaussian kernel/bandwidth as the
# circulation map. A plain kernel sum would let one looping drifter count
# many times, so each drifter counts at most once at each location: it
# contributes the kernel weight of its CLOSEST point (1 if it passed right
# through, about 0.61 at one bandwidth away, 0 beyond the cutoff). The sum
# over drifters is an "effective number of drifters nearby".

def effective_drifters(target_lat, target_lon, pts, bandwidth_km, cutoff=3.0):
    d = haversine_km(target_lat, target_lon,
                     pts["drifter_lat"].values, pts["drifter_lon"].values)
    w = np.exp(-0.5 * (d / bandwidth_km) ** 2)
    w[d > cutoff * bandwidth_km] = 0.0          # same cutoff as kernel_smooth

    # Closest point per drifter = largest weight per drifter, then sum
    codes, _ = pd.factorize(pts["drifter_id"])
    per_drifter_max = pd.DataFrame(w.T).groupby(codes).max()
    # copy=True: pandas 3 hands back a read-only array otherwise, and the
    # land mask below writes into it
    return per_drifter_max.sum(axis=0).to_numpy(copy=True)


drifter_density = {}
for season in ["Southwest", "Northeast"]:
    sp = points[points["monsoon"] == season]
    n_eff = effective_drifters(t_lat_2d.ravel(), t_lon_2d.ravel(),
                               sp, BANDWIDTH_KM).reshape(t_lat_2d.shape)
    n_eff[is_land] = np.nan
    drifter_density[season] = n_eff
    print(f"\nEffective drifters nearby — {season}: "
          f"median {np.nanmedian(n_eff):.1f}, max {np.nanmax(n_eff):.1f}")

# Cap the colour scale at the 98th percentile (as in 14.6): the SW hotspot
# near 16-17N, 88E is far above everything else and would wash out the rest
dens_max = np.nanpercentile(
    np.concatenate([drifter_density[s].ravel() for s in drifter_density]), 98)
dens_norm = Normalize(vmin=0, vmax=dens_max)
dens_cmap = plt.get_cmap("YlGnBu")

fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), constrained_layout=True,
                         subplot_kw={"projection": ccrs.PlateCarree()})

for ax, season, letter in zip(axes, ["Southwest", "Northeast"], ["a", "b"]):
    ax.set_extent([78, 99, 5, 23], crs=ccrs.PlateCarree())
    ax.pcolormesh(t_lon_2d, t_lat_2d, drifter_density[season],
                  cmap=dens_cmap, norm=dens_norm, shading="auto",
                  transform=ccrs.PlateCarree(), zorder=1)

    # Outline the area greyed out on the 14.6 map (< 3 drifters within
    # one bandwidth), so the two figures can be read together
    ax.contour(t_lon_2d, t_lat_2d,
               smoothed[season]["low_support"].astype(float), levels=[0.5],
               colors="black", linewidths=0.8, linestyles="--",
               transform=ccrs.PlateCarree(), zorder=3)

    ax.add_feature(cfeature.LAND, facecolor="#bdbdbd", zorder=2)
    ax.coastlines("50m", linewidth=0.6, zorder=2)
    gl = ax.gridlines(draw_labels=True, linewidth=0.3, alpha=0.4)
    gl.top_labels = gl.right_labels = False
    ax.set_title(f"({letter}) {season} monsoon", fontsize=13)

cbar = fig.colorbar(plt.cm.ScalarMappable(norm=dens_norm, cmap=dens_cmap),
                    ax=axes, location="right", shrink=0.8, pad=0.02,
                    extend="max")
cbar.set_label("Effective number of drifters nearby")

fig.suptitle(f"Kernel-smoothed unique-drifter coverage "
             f"(drogued only, {BANDWIDTH_KM:.0f} km bandwidth)", fontsize=15)
# Caption idea: "Each drifter counts once, weighted by how close it came
# (Gaussian kernel). Dashed line: edge of the area with fewer than 3
# drifters within one bandwidth (grey on the circulation map)."

plt.show()
