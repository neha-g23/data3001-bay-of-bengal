import os
import clouddrift as cd
from clouddrift.ragged import subset
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import Normalize
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
# 2. Observation density map (spatial coverage)
# ============================================================
# Raw observation counts per grid cell. Note: this reflects sampling
# effort as much as the underlying circulation, so it's not on its own
# evidence of "good coverage" — see the unique-drifter map further down.

print("DENSITY MAP/SPATIAL COVERAGE\n")

lon = ds_bob.lon.values
lat = ds_bob.lat.values

counts, xedges, yedges = np.histogram2d(
    lon,
    lat,
    bins=50
)

plot_counts = counts.copy()
plot_counts[plot_counts == 0] = np.nan

cmap = plt.cm.YlOrBr.copy()
cmap.set_bad("lightblue")

plt.figure(figsize=(10, 8))

im = plt.pcolormesh(
    xedges,
    yedges,
    plot_counts.T,
    cmap=cmap,
    shading="auto"
)

plt.colorbar(im, label="Number of Observations per Cell")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Drifter Observation Density: Bay of Bengal")

plt.show()


# ============================================================
# 3. Seasonal coverage (observations per month)
# ============================================================

print("SEASONAL COVERAGE\n")
df_time = ds_bob[["time"]].to_dataframe().reset_index()

df_time["month"] = pd.to_datetime(df_time["time"]).dt.month

monthly_counts = df_time["month"].value_counts().sort_index()
print(monthly_counts)

plt.figure(figsize=(10, 5))
monthly_counts.plot(kind="bar", color="steelblue")
plt.xlabel("Month")
plt.ylabel("Number of Observations")
plt.title("Seasonal Coverage: Observations per Month — Bay of Bengal")
plt.xticks(rotation=0)
# plt.show()


# ============================================================
# 4. Data quality checks
# ============================================================
# Core measurements + uncertainty + quality flags + drogue status.

cols = [
    "time", "lat", "lon",
    "ve", "vn", "err_ve", "err_vn",
    "sst", "sst1", "sst2", "err_sst", "err_sst1", "err_sst2",
    "flg_sst", "flg_sst1", "flg_sst2",
    "drogue_status", "gap"
]

df = ds_bob[cols].to_dataframe()

missing_counts = df.isna().sum()
print("Missing value counts:\n", missing_counts)

missing_pct = (df.isna().sum() / len(df)) * 100
print("\nMissing value %:\n", missing_pct)

# Check for suspicious fill values (e.g. -999) or unrealistic ranges
print("\nSummary stats:\n", df.describe())

print("\nflg_sst counts:\n", df["flg_sst"].value_counts())
print("\ndrogue_status counts:\n", df["drogue_status"].value_counts())


# ============================================================
# 5. Monsoon labeling + observation coverage by monsoon
# ============================================================

print("\n\nMONSOON\n")
df_time = ds_bob[["time"]].to_dataframe().reset_index()
df_time["month"] = pd.to_datetime(df_time["time"]).dt.month

def get_monsoon(month):
    if month in [6, 7, 8, 9]:
        return "Southwest Monsoon (Jun-Sep)"
    elif month in [12, 1, 2]:
        return "Northeast Monsoon (Dec-Feb)"
    else:
        return "Inter-monsoon (Mar-May, Oct-Nov)"

df_time["monsoon"] = df_time["month"].apply(get_monsoon)

monsoon_counts = df_time["monsoon"].value_counts()
print(monsoon_counts)

plt.figure(figsize=(8, 5))
monsoon_counts.plot(kind="bar", color=["coral", "steelblue", "lightgray"])
plt.ylabel("Number of Observations")
plt.title("Observation Coverage by Monsoon Period — Bay of Bengal")
plt.xticks(rotation=15)
# plt.show()


# ============================================================
# 6. Observations per year
# ============================================================

df_time["year"] = pd.to_datetime(df_time["time"]).dt.year

yearly_counts = df_time["year"].value_counts().sort_index()

print(yearly_counts)

plt.figure(figsize=(10, 5))
yearly_counts.plot(kind="bar")

plt.xlabel("Year")
plt.ylabel("Number of Observations")
plt.title("Bay of Bengal Drifter Observations by Year")

plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()


# ============================================================
# 7. Drogue status: overall and by monsoon
# ============================================================
# drogue_status = True means the drifter's sea anchor was still attached;
# once lost, the buoy becomes more wind-driven than current-driven.

print(df["drogue_status"].value_counts(dropna=False))

drogue_counts = df["drogue_status"].value_counts(dropna=False)
drogue_pct = drogue_counts / len(df) * 100

print(drogue_counts)
print(drogue_pct)

df["month"] = pd.to_datetime(df["time"]).dt.month
df["monsoon"] = df["month"].apply(get_monsoon)

print(pd.crosstab(df["monsoon"], df["drogue_status"]))

print(
    pd.crosstab(
        df["monsoon"],
        df["drogue_status"],
        normalize="index"
    ) * 100
)

# Percentage of drogued vs non-drogued observations within each monsoon period
drogue_by_monsoon = pd.crosstab(
    df["monsoon"],
    df["drogue_status"],
    normalize="index"
) * 100

drogue_by_monsoon = drogue_by_monsoon.rename(
    columns={
        False: "Non-drogued",
        True: "Drogued"
    }
)

drogue_by_monsoon.index = [
    "Inter-monsoon",
    "Northeast",
    "Southwest"
]

ax = drogue_by_monsoon.plot(
    kind="bar",
    stacked=True,
    figsize=(7, 4)
)

plt.ylabel("Percentage of Observations (%)")
plt.xlabel("Monsoon Period")
plt.title("Drogue Status by Monsoon Period")
plt.xticks(rotation=0)
plt.legend(title="Drogue Status", bbox_to_anchor=(1.02, 1))
plt.tight_layout()

plt.show()


# ============================================================
# 8. Drogue effect on current speed (diagnostic)
# ============================================================
# Quick check before filtering: does losing the drogue visibly change the
# current signal? If undrogued speeds run higher/more spread out, that
# supports treating undrogued velocities as wind-contaminated rather than
# a clean current measurement — the justification for filtering in
# section 13.

print("\nDROGUE EFFECT ON CURRENT SPEED\n")

speed_df = ds_bob[["ve", "vn", "drogue_status"]].to_dataframe().dropna(
    subset=["ve", "vn"]
)
speed_df["speed"] = np.sqrt(speed_df["ve"]**2 + speed_df["vn"]**2)

print(
    speed_df.groupby("drogue_status")["speed"]
    .describe()[["count", "mean", "std", "50%", "max"]]
)

ax = speed_df.boxplot(
    column="speed",
    by="drogue_status",
    showfliers=False,
    figsize=(7, 5)
)
ax.set_xlabel("Drogue Attached")
ax.set_ylabel("Current Speed (m/s)")
ax.set_title("Current Speed by Drogue Status")
plt.suptitle("")
plt.show()


# ============================================================
# 9. Trajectory map
# ============================================================
# Walks the ragged array by rowsize to plot each drifter's full track,
# marking its first fix (entry point into the region) in red.

print("TRAJECTORY MAP\n")

rowsize = ds_bob["rowsize"].values

lon = ds_bob["lon"].values
lat = ds_bob["lat"].values

plt.figure(figsize=(10, 8))

start = 0

for n in rowsize:
    end = start + int(n)

    traj_lon = lon[start:end]
    traj_lat = lat[start:end]

    valid = np.isfinite(traj_lon) & np.isfinite(traj_lat)

    traj_lon = traj_lon[valid]
    traj_lat = traj_lat[valid]

    if len(traj_lon) > 0:
        plt.plot(
            traj_lon,
            traj_lat,
            linewidth=0.4,
            alpha=0.12,
            color="blue"
        )

        plt.scatter(
            traj_lon[0],
            traj_lat[0],
            s=18,
            color="red",
            alpha=0.8
        )

    start = end

plt.xlim(78, 99)
plt.ylim(5, 23)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Drifter Trajectories and Regional Entry Points: Bay of Bengal")

plt.grid(alpha=0.2)
plt.tight_layout()
# plt.show()


# ============================================================
# 10. Naive seasonal circulation map (Southwest vs Northeast)
# ============================================================
# "Naive" because it averages ve/vn per grid cell directly, without
# weighting by drifter — a cell full of one looping drifter counts the
# same as a cell sampled by many. See section 13 for the corrected,
# drifter-weighted, drogue-filtered version.

print("\nNAIVE MONSOON CIRCULATION MAPS\n")

# --------------------------------------------------
# 10.1 Build a dataframe with the variables we need
# --------------------------------------------------

currents = ds_bob[
    ["time", "lat", "lon", "ve", "vn"]
].to_dataframe().reset_index()

currents = currents.dropna(
    subset=["time", "lat", "lon", "ve", "vn"]
)

# --------------------------------------------------
# 10.2 Label each observation by monsoon
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

# Keep only the two monsoons we are comparing
currents = currents[
    currents["monsoon"].isin(["Southwest", "Northeast"])
]

# --------------------------------------------------
# 10.3 Create a 1° x 1° spatial grid
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

# --------------------------------------------------
# 10.4 Average current in each grid cell
# --------------------------------------------------

grid = (
    currents
    .groupby(
        ["monsoon", "lat_bin", "lon_bin"],
        observed=True
    )
    .agg(
        mean_ve=("ve", "mean"),
        mean_vn=("vn", "mean"),
        n_obs=("ve", "count")
    )
    .reset_index()
)

grid["lat_center"] = grid["lat_bin"].apply(
    lambda x: x.mid
).astype(float)

grid["lon_center"] = grid["lon_bin"].apply(
    lambda x: x.mid
).astype(float)

# --------------------------------------------------
# 10.5 Only show cells with enough observations
# --------------------------------------------------

min_obs = 20

grid = grid[grid["n_obs"] >= min_obs]

print("\nCells remaining after minimum observation threshold:")
print(grid.groupby("monsoon").size())

# --------------------------------------------------
# 10.6 Plot Southwest vs Northeast
# --------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 7),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

vmax = 0

# Find common max speed across both monsoons, for a shared color scale
for season in ["Southwest", "Northeast"]:
    season_data = grid[grid["monsoon"] == season].copy()
    speed = np.sqrt(season_data["mean_ve"]**2 + season_data["mean_vn"]**2)
    vmax = max(vmax, speed.max())

q = None

for ax, season in zip(axes, ["Southwest", "Northeast"]):
    season_data = grid[grid["monsoon"] == season].copy()

    speed = np.sqrt(season_data["mean_ve"]**2 + season_data["mean_vn"]**2)
    season_data = season_data[speed > 0].copy()
    speed = speed[speed > 0]

    season_data["u_norm"] = season_data["mean_ve"] / speed
    season_data["v_norm"] = season_data["mean_vn"] / speed

    q = ax.quiver(
        season_data["lon_center"],
        season_data["lat_center"],
        season_data["u_norm"],
        season_data["v_norm"],
        speed,
        cmap="turbo",
        angles="xy",
        scale=25,
        width=0.003,
        alpha=0.9
    )

    ax.set_xlim(78, 99)
    ax.set_ylim(5, 23)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{season} Monsoon")
    ax.grid(alpha=0.2)

cbar = fig.colorbar(
    q,
    ax=axes,
    location="right",
    shrink=0.85,
    pad=0.03
)
cbar.set_label("Mean Current Speed")

fig.suptitle("Mean Surface Current Direction and Speed: Bay of Bengal", fontsize=16)

# plt.show()


# ============================================================
# 11. Observations and unique drifters per monsoon
# ============================================================

print("\nOBSERVATIONS AND DRIFTERS PER MONSOON\n")

# --------------------------------------------------
# 11.1 Give every observation its drifter ID
# --------------------------------------------------
# rowsize gives the number of observations belonging to each drifter, and
# observations are stored contiguously in drifter order, so repeating each
# drifter id by its rowsize attaches it correctly to every observation.

drifter_ids = np.repeat(
    ds_bob["id"].values,
    ds_bob["rowsize"].values.astype(int)
)

print("Observation rows:", ds_bob.sizes["obs"])
print("Drifter ID rows:", len(drifter_ids))

# --------------------------------------------------
# 11.2 Build dataframe and label monsoon period
# --------------------------------------------------

season_df = pd.DataFrame({
    "time": ds_bob["time"].values,
    "drifter_id": drifter_ids
})

season_df["month"] = pd.to_datetime(
    season_df["time"]
).dt.month

def get_monsoon(month):
    if month in [6, 7, 8, 9]:
        return "Southwest Monsoon"
    elif month in [12, 1, 2]:
        return "Northeast Monsoon"
    else:
        return "Inter-monsoon"

season_df["monsoon"] = season_df["month"].apply(get_monsoon)

# --------------------------------------------------
# 11.3 Count observations AND unique drifters per monsoon
# --------------------------------------------------

monsoon_summary = (
    season_df
    .groupby("monsoon")
    .agg(
        observations=("time", "count"),
        unique_drifters=("drifter_id", "nunique")
    )
)

print(monsoon_summary)


# ============================================================
# 12. Unique drifters per grid cell (coverage map)
# ============================================================
# Counts distinct drifters rather than raw observations, since a single
# looping drifter can inflate observation counts without adding real
# independent coverage. Uses the same 50x50 grid as the density map.

print("\nUNIQUE DRIFTERS PER GRID CELL\n")

drifter_ids = np.repeat(
    ds_bob["id"].values,
    ds_bob["rowsize"].values.astype(int)
)

drifter_df = pd.DataFrame({
    "lon": ds_bob["lon"].values,
    "lat": ds_bob["lat"].values,
    "drifter_id": drifter_ids
})

# xedges/yedges came from the density map's histogram2d(lon, lat, bins=50)
drifter_df["lon_bin"] = pd.cut(
    drifter_df["lon"],
    bins=xedges,
    include_lowest=True,
    labels=False
)

drifter_df["lat_bin"] = pd.cut(
    drifter_df["lat"],
    bins=yedges,
    include_lowest=True,
    labels=False
)

unique_counts = (
    drifter_df
    .groupby(["lon_bin", "lat_bin"])["drifter_id"]
    .nunique()
)

unique_grid = np.zeros(
    (len(xedges) - 1, len(yedges) - 1)
)

for (lon_bin, lat_bin), count in unique_counts.items():
    unique_grid[int(lon_bin), int(lat_bin)] = count

plot_unique = unique_grid.copy()

# Cells with no drifters plot as NaN (shown grey) rather than 0
plot_unique[plot_unique == 0] = np.nan

cmap = plt.cm.YlGnBu.copy()
cmap.set_bad("lightgrey")

plt.figure(figsize=(10, 8))

im = plt.pcolormesh(
    xedges,
    yedges,
    plot_unique.T,
    cmap=cmap,
    shading="auto"
)

plt.colorbar(
    im,
    label="Number of Unique Drifters per Cell"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Unique Drifter Coverage: Bay of Bengal")

plt.show()


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
            drifter_obs=("ve", "count")
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
# 13.5 Coverage threshold sensitivity check — run separately for each
#      subset, since drogue-filtering removes drifters and can push
#      cells that cleared the threshold on all-data below it.
# --------------------------------------------------

min_drifters_options = [1, 3, 5, 10]

def apply_coverage_threshold(cell_grid, chosen_min_drifters, label):
    print(f"\nCOVERAGE THRESHOLD SENSITIVITY — {label}\n")

    sensitivity_rows = []
    for candidate in min_drifters_options:
        retained = cell_grid[cell_grid["n_drifters"] >= candidate]
        counts = retained.groupby("monsoon").size()
        sensitivity_rows.append({
            "min_drifters": candidate,
            "Southwest_cells": counts.get("Southwest", 0),
            "Northeast_cells": counts.get("Northeast", 0),
        })
    print(pd.DataFrame(sensitivity_rows).set_index("min_drifters"))

    filtered = cell_grid[cell_grid["n_drifters"] >= chosen_min_drifters].copy()
    below_threshold = cell_grid[cell_grid["n_drifters"] < chosen_min_drifters].copy()

    print(f"\nCells retained at min_drifters = {chosen_min_drifters} ({label}):")
    print(
        filtered.groupby("monsoon")
        .agg(
            cells=("mean_ve", "size"),
            mean_drifters_per_cell=("n_drifters", "mean"),
            min_drifters=("n_drifters", "min"),
            max_drifters=("n_drifters", "max")
        )
    )
    return filtered, below_threshold

# Chosen threshold — revisit if either sensitivity table above shows a
# steep drop-off before this point (see CLAUDE.md "Sampling-bias handling").
min_drifters = 3

grid_filtered_all, insufficient_all = apply_coverage_threshold(
    grid_all, min_drifters, "ALL DATA"
)
grid_filtered_drogued, insufficient_drogued = apply_coverage_threshold(
    grid_drogued, min_drifters, "DROGUED ONLY"
)

# --------------------------------------------------
# 13.6 Calculate speed + normalized direction
# --------------------------------------------------

def add_speed_and_direction(filtered_grid):
    filtered_grid = filtered_grid.copy()
    filtered_grid["speed"] = np.sqrt(
        filtered_grid["mean_ve"]**2 + filtered_grid["mean_vn"]**2
    )
    filtered_grid = filtered_grid[filtered_grid["speed"] > 0].copy()
    filtered_grid["u_norm"] = filtered_grid["mean_ve"] / filtered_grid["speed"]
    filtered_grid["v_norm"] = filtered_grid["mean_vn"] / filtered_grid["speed"]
    return filtered_grid

grid_filtered_all = add_speed_and_direction(grid_filtered_all)
grid_filtered_drogued = add_speed_and_direction(grid_filtered_drogued)

# --------------------------------------------------
# 13.7 Plot: rows = all data / drogued-only, columns = SW / NE
# --------------------------------------------------

def plot_season_quiver(ax, season_data, season_insufficient, norm, cmap, title):
    # Shade sampled-but-below-threshold cells grey/hatched rather than
    # leaving them blank, so "insufficient data" reads as a deliberate
    # result rather than a gap in the plot.
    for _, cell in season_insufficient.iterrows():
        ax.add_patch(Rectangle(
            (cell["lon_center"] - grid_size / 2, cell["lat_center"] - grid_size / 2),
            grid_size,
            grid_size,
            facecolor="lightgrey",
            edgecolor="none",
            hatch="//",
            alpha=0.5,
            zorder=0
        ))

    # Colour each cell's background by speed, so direction (the arrow)
    # and magnitude (the cell colour) read as two separate channels.
    for _, cell in season_data.iterrows():
        ax.add_patch(Rectangle(
            (cell["lon_center"] - grid_size / 2, cell["lat_center"] - grid_size / 2),
            grid_size,
            grid_size,
            facecolor=cmap(norm(cell["speed"])),
            edgecolor="none",
            zorder=0
        ))

    ax.quiver(
        season_data["lon_center"],
        season_data["lat_center"],
        season_data["u_norm"],
        season_data["v_norm"],
        color="black",
        angles="xy",
        scale=25,
        width=0.003
    )

    ax.set_xlim(78, 99)
    ax.set_ylim(5, 23)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    ax.grid(alpha=0.2)

fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 7),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

norm = Normalize(vmin=grid_filtered_drogued["speed"].min(), vmax=grid_filtered_drogued["speed"].max())
cmap = plt.get_cmap("YlGnBu")

for ax, season in zip(axes, ["Southwest", "Northeast"]):
    season_data = grid_filtered_drogued[grid_filtered_drogued["monsoon"] == season]
    season_insufficient = insufficient_drogued[insufficient_drogued["monsoon"] == season]

    plot_season_quiver(
        ax, season_data, season_insufficient, norm, cmap,
        f"Drogued Only — {season} Monsoon"
    )

insufficient_patch = Rectangle(
    (0, 0), 1, 1,
    facecolor="lightgrey",
    edgecolor="none",
    hatch="//",
    alpha=0.5,
    label=f"Insufficient data (<{min_drifters} drifters with ≥{MIN_OBS_PER_DRIFTER} obs)"
)
axes[0].legend(handles=[insufficient_patch], loc="upper left", fontsize=8)

cbar = fig.colorbar(
    plt.cm.ScalarMappable(norm=norm, cmap=cmap),
    ax=axes,
    location="right",
    shrink=0.8,
    pad=0.03
)

cbar.set_label("Mean Current Speed")

fig.suptitle(
    "Drifter-Weighted Mean Surface Circulation: Drogued Only",
    fontsize=16
)

plt.show()


# Build observation table with drifter id and drogue status
ids = np.repeat(ds_bob["id"].values, ds_bob["rowsize"].values.astype(int))
cur = pd.DataFrame({
    "time": ds_bob["time"].values,
    "lat": ds_bob["lat"].values,
    "lon": ds_bob["lon"].values,
    "ve": ds_bob["ve"].values,
    "vn": ds_bob["vn"].values,
    "drogue": ds_bob["drogue_status"].values,
    "drifter_id": ids,
}).dropna(subset=["time", "lat", "lon", "ve", "vn"])

month = pd.to_datetime(cur["time"]).dt.month
cur["monsoon"] = np.where(month.isin([6, 7, 8, 9]), "SW",
                  np.where(month.isin([12, 1, 2]), "NE", "other"))
cur = cur[cur["monsoon"].isin(["SW", "NE"])].copy()

# 1-degree cells
cur["lat_bin"] = np.floor(cur["lat"])
cur["lon_bin"] = np.floor(cur["lon"])

def drifter_weighted(d, min_drifters=3):
    # average within each drifter first, then across drifters
    per = (d.groupby(["monsoon", "lat_bin", "lon_bin", "drifter_id"])
             .agg(ve=("ve", "mean"), vn=("vn", "mean")).reset_index())
    g = (per.groupby(["monsoon", "lat_bin", "lon_bin"])
            .agg(ve=("ve", "mean"), vn=("vn", "mean"), n=("drifter_id", "nunique"))
            .reset_index())
    return g[g["n"] >= min_drifters]

dr = drifter_weighted(cur[cur["drogue"]])     # drogued only
un = drifter_weighted(cur[~cur["drogue"]])    # undrogued only
allg = drifter_weighted(cur)                  # everything

m = dr.merge(un, on=["monsoon", "lat_bin", "lon_bin"], suffixes=("_dr", "_un"))
ang_dr = np.degrees(np.arctan2(m["vn_dr"], m["ve_dr"]))
ang_un = np.degrees(np.arctan2(m["vn_un"], m["ve_un"]))
m["angle_diff"] = np.abs((ang_un - ang_dr + 180) % 360 - 180)   # 0 to 180 degrees

print("Shared cells per season:\n", m.groupby("monsoon").size())
print(m.groupby("monsoon")["angle_diff"].describe())

def reversed_fraction(g):
    sw = g[g["monsoon"] == "SW"].set_index(["lat_bin", "lon_bin"])
    ne = g[g["monsoon"] == "NE"].set_index(["lat_bin", "lon_bin"])
    b = sw[["ve", "vn"]].join(ne[["ve", "vn"]], lsuffix="_sw", rsuffix="_ne", how="inner")
    dot = b["ve_sw"] * b["ve_ne"] + b["vn_sw"] * b["vn_ne"]
    return len(b), (dot < 0).mean()

# for name, g in [("drogued only", dr), ("undrogued only", un), ("all data", allg)]:
#     n, frac = reversed_fraction(g)
#     print(f"{name}: {n} cells with both seasons, {frac*100:.1f}% reversed")


rng = np.random.default_rng(0)

def reversed_frac(per, min_drifters=3):
    p = per.assign(wve=per["ve"] * per["w"], wvn=per["vn"] * per["w"])
    g = (p.groupby(["monsoon", "lat_bin", "lon_bin"])
           .agg(wve=("wve", "sum"), wvn=("wvn", "sum"), w=("w", "sum"),
                n=("drifter_id", "size")).reset_index())
    g = g[g["n"] >= min_drifters]
    g["ve"], g["vn"] = g["wve"] / g["w"], g["wvn"] / g["w"]
    sw = g[g["monsoon"] == "SW"].set_index(["lat_bin", "lon_bin"])
    ne = g[g["monsoon"] == "NE"].set_index(["lat_bin", "lon_bin"])
    b = sw[["ve", "vn"]].join(ne[["ve", "vn"]], lsuffix="_sw", rsuffix="_ne", how="inner")
    return ((b["ve_sw"] * b["ve_ne"] + b["vn_sw"] * b["vn_ne"]) < 0).mean()

def bootstrap_ci(d, n_boot=500):
    per = (d.groupby(["monsoon", "lat_bin", "lon_bin", "drifter_id"])
             .agg(ve=("ve", "mean"), vn=("vn", "mean")).reset_index())
    ids = per["drifter_id"].unique()
    out = []
    for _ in range(n_boot):
        w = pd.Series(rng.choice(ids, size=len(ids))).value_counts().rename("w")
        pb = per.merge(w, left_on="drifter_id", right_index=True)
        out.append(reversed_frac(pb))
    return np.nanpercentile(out, [2.5, 50, 97.5])

# for name, d in [("drogued only", cur[cur["drogue"]]),
#                 ("undrogued only", cur[~cur["drogue"]]),
#                 ("all data", cur)]:
#     lo, med, hi = bootstrap_ci(d)
#     print(f"{name}: {med*100:.1f}% reversed (95% CI {lo*100:.1f}-{hi*100:.1f}%)")


def bootstrap_ci(d, min_drifters=3, n_boot=500):
    per = (d.groupby(["monsoon", "lat_bin", "lon_bin", "drifter_id"])
             .agg(ve=("ve", "mean"), vn=("vn", "mean")).reset_index())
    ids = per["drifter_id"].unique()
    out = []
    for _ in range(n_boot):
        w = pd.Series(rng.choice(ids, size=len(ids))).value_counts().rename("w")
        pb = per.merge(w, left_on="drifter_id", right_index=True)
        out.append(reversed_frac(pb, min_drifters))
    return np.nanpercentile(out, [2.5, 50, 97.5])

for k in [3, 5]:
    lo, med, hi = bootstrap_ci(cur[cur["drogue"]], min_drifters=k)
    print(f"drogued only, min_drifters={k}: {med*100:.1f}% (95% CI {lo*100:.1f}-{hi*100:.1f}%)")


m = allg.merge(dr, on=["monsoon", "lat_bin", "lon_bin"], suffixes=("_all", "_dr"))

# Speed in each version
m["speed_all"] = np.sqrt(m["ve_all"]**2 + m["vn_all"]**2)
m["speed_dr"] = np.sqrt(m["ve_dr"]**2 + m["vn_dr"]**2)

# Vector difference magnitude (how far apart the two arrows are, in m/s)
m["vec_diff"] = np.sqrt((m["ve_all"] - m["ve_dr"])**2 + (m["vn_all"] - m["vn_dr"])**2)

# As a percentage of the all-data speed, so it's comparable across fast/slow regions
m["pct_diff"] = m["vec_diff"] / m["speed_all"] * 100

print(m.groupby("monsoon")[["vec_diff", "pct_diff"]].describe())


print("Speed correlation (all-data vs drogued-only):", m["speed_all"].corr(m["speed_dr"]))

# --------------------------------------------------
# 1. Merge the two grids on shared cells
# --------------------------------------------------

m = allg.merge(dr, on=["monsoon", "lat_bin", "lon_bin"], suffixes=("_all", "_dr"))

print(f"Shared cells: {len(m)}")
print(m.groupby("monsoon").size())

# --------------------------------------------------
# 2. Speed in each version, and % speed difference
# --------------------------------------------------

m["speed_all"] = np.sqrt(m["ve_all"]**2 + m["vn_all"]**2)
m["speed_dr"]  = np.sqrt(m["ve_dr"]**2  + m["vn_dr"]**2)

m["pct_diff"] = np.abs(m["speed_all"] - m["speed_dr"]) / m["speed_all"] * 100

# --------------------------------------------------
# 3. Direction difference (angle between the two vectors)
# --------------------------------------------------

m["angle_all"] = np.degrees(np.arctan2(m["vn_all"], m["ve_all"]))
m["angle_dr"]  = np.degrees(np.arctan2(m["vn_dr"],  m["ve_dr"]))
m["angle_diff"] = np.abs((m["angle_dr"] - m["angle_all"] + 180) % 360 - 180)

print("\nSpeed and direction difference summary:")
print(m.groupby("monsoon")[["pct_diff", "angle_diff"]].describe())

# --------------------------------------------------
# 4. Overall speed correlation (all-data vs drogued-only)
# --------------------------------------------------

print("\nSpeed correlation (all-data vs drogued-only):",
      m["speed_all"].corr(m["speed_dr"]))

# --------------------------------------------------
# 5. Does the difference shrink with more drogued drifters?
#    (rules out "difference = just a noisy small sample")
# --------------------------------------------------

print("\nCorrelation: n_drogued_drifters vs % speed difference:",
      m["n_dr"].corr(m["pct_diff"]))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].scatter(m["n_dr"], m["pct_diff"], alpha=0.5)
axes[0].set_xlabel("Number of drogued drifters in cell")
axes[0].set_ylabel("% speed difference")
axes[0].set_title("Speed difference vs drogued sample size")

axes[1].scatter(m["n_dr"], m["angle_diff"], alpha=0.5, color="orange")
axes[1].set_xlabel("Number of drogued drifters in cell")
axes[1].set_ylabel("Angle difference (degrees)")
axes[1].set_title("Direction difference vs drogued sample size")

plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6. Map of where the differences are largest
# --------------------------------------------------

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sc1 = axes[0].scatter(m["lon_bin"], m["lat_bin"], c=m["pct_diff"],
                       cmap="RdYlGn_r", vmax=50)
plt.colorbar(sc1, ax=axes[0], label="% speed difference")
axes[0].set_title("Speed difference: all-data vs drogued-only")
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")

sc2 = axes[1].scatter(m["lon_bin"], m["lat_bin"], c=m["angle_diff"],
                       cmap="RdYlGn_r", vmax=90)
plt.colorbar(sc2, ax=axes[1], label="Direction difference (degrees)")
axes[1].set_title("Direction difference: all-data vs drogued-only")
axes[1].set_xlabel("Longitude")
axes[1].set_ylabel("Latitude")

plt.tight_layout()
plt.show()

# Correlation with everything
# print("With outliers:", m["n_dr"].corr(m["pct_diff"]))

# Correlation after removing the most extreme pct_diff values
m_trimmed = m[m["pct_diff"] < m["pct_diff"].quantile(0.95)]  # drop top 5%
# print("Without top 5% outliers:", m_trimmed["n_dr"].corr(m_trimmed["pct_diff"]))

# for min_n in [3, 10, 15, 20, 25]:
#     subset = m[m["n_dr"] >= min_n]
#     print(f"\nmin_drifters = {min_n}: {len(subset)} cells")
#     print(subset.groupby("monsoon")[["pct_diff", "angle_diff"]].median())

speed = np.sqrt(currents["ve"]**2 + currents["vn"]**2)

# for t in [1.5, 2.0, 2.5, 3.0, 3.5]:
#     n = (speed > t).sum()
#     print(f"Threshold {t} m/s: {n} rows above ({n/len(currents)*100:.3f}%)")

def bootstrap_cell_uncertainty(drifter_cell, n_boot=500):
    results = []
    
    for (monsoon, lat_bin, lon_bin), group in drifter_cell.groupby(
        ["monsoon", "lat_bin", "lon_bin"], observed=True
    ):
        drifter_ve = group["drifter_ve"].values
        drifter_vn = group["drifter_vn"].values
        n = len(drifter_ve)
        
        if n < 3:
            continue  # skip cells below your threshold anyway
        
        boot_ve = np.array([
            np.mean(rng.choice(drifter_ve, size=n, replace=True))
            for _ in range(n_boot)
        ])
        boot_vn = np.array([
            np.mean(rng.choice(drifter_vn, size=n, replace=True))
            for _ in range(n_boot)
        ])
        
        results.append({
            "monsoon": monsoon, "lat_bin": lat_bin, "lon_bin": lon_bin,
            "mean_ve": drifter_ve.mean(), "mean_vn": drifter_vn.mean(),
            "se_ve": boot_ve.std(), "se_vn": boot_vn.std(),
            "n_drifters": n
        })
    
    return pd.DataFrame(results)

rng = np.random.default_rng(0)
uncertainty_grid = bootstrap_cell_uncertainty(drifter_cell_drogued)
def confidence_tier(n):
    if n < 3:
        return "insufficient"
    elif n < 10:
        return "low confidence"
    else:
        return "high confidence"

uncertainty_grid["tier"] = uncertainty_grid["n_drifters"].apply(confidence_tier)

# print(uncertainty_grid.head(20))
# print(f"\nTotal cells with uncertainty estimates: {len(uncertainty_grid)}")
# print(uncertainty_grid.groupby(["monsoon", "tier"]).size())

# print(currents.columns.tolist())  # confirm what's actually in there right now, for reference

# Add error columns straight from ds_bob (same length/order as your other observation arrays)
currents_err = pd.DataFrame({
    "err_ve": ds_bob["err_ve"].values,
    "err_vn": ds_bob["err_vn"].values,
})

# print(currents_err.describe())

for t in [0.1, 0.2, 0.5, 1.0]:
    n = (currents_err["err_ve"] > t).sum()
    print(f"err_ve > {t}: {n} rows ({n/len(currents_err)*100:.3f}%)")

# Quick version — average err_ve per cell, compare to your bootstrap se_ve
err_by_obs = pd.DataFrame({
    "lat": ds_bob["lat"].values,
    "lon": ds_bob["lon"].values,
    "err_ve": ds_bob["err_ve"].values,
})

grid_size = 1.0
lat_bins = np.arange(5, 23 + grid_size, grid_size)
lon_bins = np.arange(78, 99 + grid_size, grid_size)

err_by_obs["lat_bin"] = pd.cut(err_by_obs["lat"], bins=lat_bins, include_lowest=True)
err_by_obs["lon_bin"] = pd.cut(err_by_obs["lon"], bins=lon_bins, include_lowest=True)

cell_err = err_by_obs.groupby(["lat_bin", "lon_bin"], observed=True)["err_ve"].mean().reset_index()
cell_err.columns = ["lat_bin", "lon_bin", "avg_obs_err_ve"]

check = uncertainty_grid.merge(cell_err, on=["lat_bin", "lon_bin"], how="left")
# print(check[["se_ve", "avg_obs_err_ve"]].describe())
# print("Correlation:", check["se_ve"].corr(check["avg_obs_err_ve"]))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# APPROXIMATE EICC / WESTERN BAY ANALYSIS
# ============================================================

# IMPORTANT:
# This is only an approximate EICC region.
# It is a longitude band in the western Bay, not a precise
# coastline-following EICC mask.

EICC_LON_MIN = 80.0
EICC_LON_MAX = 85.0

MIN_DRIFTERS = 3
N_BOOT = 500


# ============================================================
# 1. SELECT THE APPROXIMATE EICC BAND FROM EXISTING GRIDS
# ============================================================

def in_eicc_band(grid, lon_min=EICC_LON_MIN, lon_max=EICC_LON_MAX):

    # Use the already-created grid-cell centre if available
    if "lon_center" in grid.columns:
        lon = grid["lon_center"]

    else:
        lon = grid["lon_bin"].apply(
            lambda x: x.mid if hasattr(x, "mid") else x
        ).astype(float)

    return grid[
        (lon >= lon_min) &
        (lon <= lon_max)
    ].copy()


# These assume you already created:
# grid_filtered_all
# grid_filtered_drogued

eicc_all = in_eicc_band(grid_filtered_all)
eicc_drogued = in_eicc_band(grid_filtered_drogued)


print("\nEICC / WESTERN BAY GRID COVERAGE")
print("--------------------------------")

print("\nAll observations:")
print(eicc_all.groupby("monsoon").size())

print("\nDrogued observations only:")
print(eicc_drogued.groupby("monsoon").size())


# ============================================================
# 2. PLOT DROGUED-ONLY MEAN CURRENT FIELD
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10, 9),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

# Use the same speed colour scale for BOTH monsoons
if len(eicc_drogued) > 0:
    vmin = eicc_drogued["speed"].min()
    vmax = eicc_drogued["speed"].max()
else:
    vmin = 0
    vmax = 1


q = None

for ax, season in zip(
    axes,
    ["Southwest", "Northeast"]
):

    season_data = eicc_drogued[
        eicc_drogued["monsoon"] == season
    ]

    q = ax.quiver(
        season_data["lon_center"],
        season_data["lat_center"],

        # normalized vectors:
        # direction shown by arrow, not arrow length
        season_data["u_norm"],
        season_data["v_norm"],

        # colour represents speed
        season_data["speed"],

        cmap="plasma",
        clim=(vmin, vmax),

        angles="xy",
        scale=15,
        width=0.005
    )

    ax.set_xlim(EICC_LON_MIN, EICC_LON_MAX)
    ax.set_ylim(5, 23)

    ax.set_title(
        f"Approx. EICC Band — {season} Monsoon"
    )

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.grid(alpha=0.2)


if q is not None:
    cbar = fig.colorbar(
        q,
        ax=axes,
        location="right",
        shrink=0.8,
        pad=0.03
    )

    cbar.set_label("Mean Current Speed")

fig.suptitle(
    "Drifter-Weighted Surface Circulation in the Approximate EICC Band",
    fontsize=15
)

plt.show()


# ============================================================
# 3. HELPER:
# BUILD ONE MEAN VECTOR PER GRID CELL + MONSOON
# ============================================================

def build_eicc_cell_means(
    per,
    min_drifters=MIN_DRIFTERS,
    lon_range=(EICC_LON_MIN, EICC_LON_MAX)
):

    p = per.copy()

    # Handle either SW/NE or Southwest/Northeast labels
    p["monsoon"] = p["monsoon"].replace({
        "Southwest": "SW",
        "Northeast": "NE",
        "Southwest Monsoon": "SW",
        "Northeast Monsoon": "NE"
    })

    # If bootstrap weights are not present,
    # every drifter gets weight 1
    if "w" not in p.columns:
        p["w"] = 1

    # Weighted ve/vn
    p["wve"] = p["ve"] * p["w"]
    p["wvn"] = p["vn"] * p["w"]

    # One mean current vector per:
    # monsoon + grid cell
    g = (
        p.groupby(
            ["monsoon", "lat_bin", "lon_bin"],
            observed=True
        )
        .agg(
            wve=("wve", "sum"),
            wvn=("wvn", "sum"),
            total_weight=("w", "sum"),

            # number of DIFFERENT drifter rows
            # contributing before bootstrap weighting
            n_drifters=("drifter_id", "nunique")
        )
        .reset_index()
    )

    # Apply minimum drifter coverage
    g = g[
        g["n_drifters"] >= min_drifters
    ].copy()

    # Weighted mean current components
    g["ve"] = g["wve"] / g["total_weight"]
    g["vn"] = g["wvn"] / g["total_weight"]

    # Longitude centre
    g["lon_center"] = g["lon_bin"].apply(
        lambda x: x.mid if hasattr(x, "mid") else x
    ).astype(float)

    # Keep only approximate EICC band
    g = g[
        (g["lon_center"] >= lon_range[0]) &
        (g["lon_center"] <= lon_range[1])
    ].copy()

    return g


# ============================================================
# 4. CALCULATE SW-vs-NE DIRECTION CHANGE
# ============================================================

def calculate_eicc_reversal(
    per,
    min_drifters=MIN_DRIFTERS,
    lon_range=(EICC_LON_MIN, EICC_LON_MAX)
):

    g = build_eicc_cell_means(
        per,
        min_drifters=min_drifters,
        lon_range=lon_range
    )

    # Separate monsoons
    sw = (
        g[g["monsoon"] == "SW"]
        .set_index(["lat_bin", "lon_bin"])
    )

    ne = (
        g[g["monsoon"] == "NE"]
        .set_index(["lat_bin", "lon_bin"])
    )

    # Keep ONLY cells that exist in BOTH monsoons
    both = sw[
        ["ve", "vn", "n_drifters"]
    ].join(
        ne[
            ["ve", "vn", "n_drifters"]
        ],
        lsuffix="_sw",
        rsuffix="_ne",
        how="inner"
    )

    if len(both) == 0:
        return np.nan, pd.DataFrame()

    # --------------------------------------------------------
    # Calculate angle between SW and NE vectors
    # --------------------------------------------------------

    dot = (
        both["ve_sw"] * both["ve_ne"] +
        both["vn_sw"] * both["vn_ne"]
    )

    speed_sw = np.sqrt(
        both["ve_sw"]**2 +
        both["vn_sw"]**2
    )

    speed_ne = np.sqrt(
        both["ve_ne"]**2 +
        both["vn_ne"]**2
    )

    denominator = speed_sw * speed_ne

    # Avoid divide-by-zero
    valid = denominator > 0

    both = both[valid].copy()
    dot = dot[valid]
    denominator = denominator[valid]

    # cos(theta)
    cos_angle = dot / denominator

    # numerical protection
    cos_angle = np.clip(
        cos_angle,
        -1,
        1
    )

    # angle difference in degrees
    both["angle_change"] = np.degrees(
        np.arccos(cos_angle)
    )

    # Define "reversed" as > 90 degree directional change
    both["reversed"] = (
        both["angle_change"] > 90
    )

    reversal_fraction = both["reversed"].mean()

    return reversal_fraction, both.reset_index()


# ============================================================
# 5. BOOTSTRAP ACROSS DRIFTERS
# ============================================================

def bootstrap_ci_eicc(
    d,
    n_boot=N_BOOT,
    seed=0,
    min_drifters=MIN_DRIFTERS
):

    # --------------------------------------------------------
    # First:
    # one mean ve/vn for each individual drifter
    # within each cell and monsoon
    # --------------------------------------------------------

    per = (
        d.groupby(
            [
                "monsoon",
                "lat_bin",
                "lon_bin",
                "drifter_id"
            ],
            observed=True
        )
        .agg(
            ve=("ve", "mean"),
            vn=("vn", "mean")
        )
        .reset_index()
    )

    ids = per["drifter_id"].unique()

    rng = np.random.default_rng(seed)

    bootstrap_results = []

    for _ in range(n_boot):

        # ----------------------------------------------------
        # Resample WHOLE DRIFTERS
        #
        # Important:
        # ve and vn remain paired because we are sampling
        # drifter IDs rather than sampling ve/vn separately.
        # ----------------------------------------------------

        sampled_ids = rng.choice(
            ids,
            size=len(ids),
            replace=True
        )

        # Count how many times each drifter was selected
        weights = (
            pd.Series(sampled_ids)
            .value_counts()
            .rename("w")
        )

        # Attach bootstrap weight to each drifter's
        # cell-level mean
        boot_per = per.merge(
            weights,
            left_on="drifter_id",
            right_index=True,
            how="inner"
        )

        frac, _ = calculate_eicc_reversal(
            boot_per,
            min_drifters=min_drifters
        )

        bootstrap_results.append(frac)

    bootstrap_results = np.array(
        bootstrap_results,
        dtype=float
    )

    bootstrap_results = bootstrap_results[
        ~np.isnan(bootstrap_results)
    ]

    # 95% percentile bootstrap interval
    lo, median, hi = np.percentile(
        bootstrap_results,
        [2.5, 50, 97.5]
    )

    return lo, median, hi, bootstrap_results


# ============================================================
# 6. ORIGINAL SAMPLE ESTIMATE
# ============================================================

# Per-drifter means for drogued data
per_drogued = (
    cur[cur["drogue"]]
    .groupby(
        [
            "monsoon",
            "lat_bin",
            "lon_bin",
            "drifter_id"
        ],
        observed=True
    )
    .agg(
        ve=("ve", "mean"),
        vn=("vn", "mean")
    )
    .reset_index()
)

drogued_point_estimate, drogued_cells = calculate_eicc_reversal(
    per_drogued
)


# Per-drifter means for ALL observations
per_all = (
    cur.groupby(
        [
            "monsoon",
            "lat_bin",
            "lon_bin",
            "drifter_id"
        ],
        observed=True
    )
    .agg(
        ve=("ve", "mean"),
        vn=("vn", "mean")
    )
    .reset_index()
)

all_point_estimate, all_cells = calculate_eicc_reversal(
    per_all
)


# ============================================================
# 7. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================

lo_d, med_d, hi_d, boot_d = bootstrap_ci_eicc(
    cur[cur["drogue"]],
    n_boot=N_BOOT,
    seed=0
)

lo_a, med_a, hi_a, boot_a = bootstrap_ci_eicc(
    cur,
    n_boot=N_BOOT,
    seed=0
)


# ============================================================
# 8. PRINT RESULTS
# ============================================================

print("\n========================================")
print("APPROXIMATE EICC BAND RESULTS")
print("========================================")

print("\nDefinition:")
print(
    "Directional reversal = SW and NE mean current "
    "vectors differ by more than 90 degrees."
)

print(
    f"\nMinimum coverage = {MIN_DRIFTERS} unique "
    "drifters per cell per monsoon."
)


print("\n--- DROGUED ONLY ---")

print(
    f"Matched SW-NE cells: "
    f"{len(drogued_cells)}"
)

print(
    f"Observed reversal fraction: "
    f"{drogued_point_estimate * 100:.1f}%"
)

print(
    f"Bootstrap median: "
    f"{med_d * 100:.1f}%"
)

print(
    f"95% bootstrap CI: "
    f"{lo_d * 100:.1f}% "
    f"to {hi_d * 100:.1f}%"
)


print("\n--- ALL OBSERVATIONS ---")

print(
    f"Matched SW-NE cells: "
    f"{len(all_cells)}"
)

print(
    f"Observed reversal fraction: "
    f"{all_point_estimate * 100:.1f}%"
)

print(
    f"Bootstrap median: "
    f"{med_a * 100:.1f}%"
)

print(
    f"95% bootstrap CI: "
    f"{lo_a * 100:.1f}% "
    f"to {hi_a * 100:.1f}%"
)


# ============================================================
# 9. OPTIONAL: INSPECT ANGLE CHANGES BY CELL
# ============================================================

# print("\nExample drogued-only matched cells:")

# print(
#     drogued_cells[
#         [
#             "lat_bin",
#             "lon_bin",
#             "ve_sw",
#             "vn_sw",
#             "ve_ne",
#             "vn_ne",
#             "angle_change",
#             "reversed"
#         ]
#     ]
#     .sort_values(
#         "angle_change",
#         ascending=False
#     )
#     .head(15)
# )

# print(
#     drogued_cells[
#         ["lat_bin", "lon_bin", "angle_change", "reversed"]
#     ].sort_values("angle_change")
# )
# print(drogued_cells["reversed"].value_counts())
# print(
#     drogued_cells["reversed"].value_counts(normalize=True) * 100
# )

def reversal_category(angle):
    if angle < 45:
        return "Little change"
    elif angle < 90:
        return "Moderate change"
    elif angle < 135:
        return "Strong directional change"
    else:
        return "Near-opposite reversal"

drogued_cells["reversal_category"] = (
    drogued_cells["angle_change"].apply(reversal_category)
)

# print(
#     drogued_cells["reversal_category"]
#     .value_counts()
# )

#####################

# ============================================================
# 1. SPEED DISTRIBUTION: DROGUED VS UNDROGUED BY MONSOON
# ============================================================

cur = cur.copy()

cur["speed"] = np.sqrt(cur["ve"]**2 + cur["vn"]**2)

cur["drogue_group"] = np.where(
    cur["drogue"],
    "Drogued",
    "Undrogued"
)

speed_summary = (
    cur.groupby(["monsoon", "drogue_group"])
    .agg(
        n_obs=("speed", "size"),
        mean_speed=("speed", "mean"),
        median_speed=("speed", "median"),
        std_speed=("speed", "std"),
        p25=("speed", lambda x: x.quantile(0.25)),
        p75=("speed", lambda x: x.quantile(0.75))
    )
)

print("\nSPEED SUMMARY")
print(speed_summary)


# Boxplot
plot_order = ["SW", "NE"]

fig, axes = plt.subplots(
    1, 2,
    figsize=(12, 5),
    sharey=True,
    constrained_layout=True
)

for ax, season in zip(axes, plot_order):

    drogued = cur[
        (cur["monsoon"] == season) &
        (cur["drogue"])
    ]["speed"].dropna()

    undrogued = cur[
        (cur["monsoon"] == season) &
        (~cur["drogue"])
    ]["speed"].dropna()

    ax.boxplot(
        [drogued, undrogued],
        labels=["Drogued", "Undrogued"],
        showfliers=False
    )

    ax.set_title(
        "Southwest Monsoon" if season == "SW"
        else "Northeast Monsoon"
    )

    ax.set_ylabel("Current Speed")

plt.suptitle(
    "Current Speed by Drogue Status and Monsoon"
)

plt.show()

# ============================================================
# 2. FUNCTION TO BUILD DRIFTER-WEIGHTED GRID
# ============================================================

def build_drifter_weighted_grid(data, min_drifters=3):

    # One mean per drifter, per cell, per monsoon
    drifter_cell = (
        data.groupby(
            ["monsoon", "lat_bin", "lon_bin", "drifter_id"],
            observed=True
        )
        .agg(
            drifter_ve=("ve", "mean"),
            drifter_vn=("vn", "mean"),
            n_obs=("ve", "count")
        )
        .reset_index()
    )

    # Then average across drifters
    grid = (
        drifter_cell.groupby(
            ["monsoon", "lat_bin", "lon_bin"],
            observed=True
        )
        .agg(
            mean_ve=("drifter_ve", "mean"),
            mean_vn=("drifter_vn", "mean"),
            n_drifters=("drifter_id", "nunique"),
            n_obs=("n_obs", "sum")
        )
        .reset_index()
    )

    grid["lat_center"] = grid["lat_bin"].apply(
        lambda x: x.mid if hasattr(x, "mid") else x
    ).astype(float)

    grid["lon_center"] = grid["lon_bin"].apply(
        lambda x: x.mid if hasattr(x, "mid") else x
    ).astype(float)

    grid = grid[
        grid["n_drifters"] >= min_drifters
    ].copy()

    grid["speed"] = np.sqrt(
        grid["mean_ve"]**2 +
        grid["mean_vn"]**2
    )

    grid = grid[grid["speed"] > 0].copy()

    grid["u_norm"] = (
        grid["mean_ve"] / grid["speed"]
    )

    grid["v_norm"] = (
        grid["mean_vn"] / grid["speed"]
    )

    return grid


grid_all = build_drifter_weighted_grid(
    cur,
    min_drifters=3
)

grid_drogued = build_drifter_weighted_grid(
    cur[cur["drogue"]],
    min_drifters=3
)

# ============================================================
# 3. ALL DATA VS DROGUED ONLY MAPS
# ============================================================

fig, axes = plt.subplots(
    2, 2,
    figsize=(14, 12),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

datasets = [
    ("All Data — SW", grid_all, "SW"),
    ("All Data — NE", grid_all, "NE"),
    ("Drogued Only — SW", grid_drogued, "SW"),
    ("Drogued Only — NE", grid_drogued, "NE")
]

combined_speed = pd.concat([
    grid_all["speed"],
    grid_drogued["speed"]
])

vmin = combined_speed.min()
vmax = combined_speed.max()

q = None

for ax, (title, grid_data, season) in zip(
    axes.flatten(),
    datasets
):

    d = grid_data[
        grid_data["monsoon"] == season
    ]

    q = ax.quiver(
        d["lon_center"],
        d["lat_center"],
        d["u_norm"],
        d["v_norm"],
        d["speed"],
        cmap="plasma",
        clim=(vmin, vmax),
        angles="xy",
        scale=25,
        width=0.003
    )

    ax.set_xlim(78, 99)
    ax.set_ylim(5, 23)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.grid(alpha=0.2)

cbar = fig.colorbar(
    q,
    ax=axes.ravel().tolist(),
    shrink=0.8,
    pad=0.02
)

cbar.set_label("Mean Current Speed")

fig.suptitle(
    "Sensitivity of Mean Circulation to Drogue Status",
    fontsize=16
)

plt.show()

# ============================================================
# 4. CELL-BY-CELL DROGUE SENSITIVITY
# ============================================================

def compare_all_vs_drogued(grid_all, grid_drogued, season):

    a = (
        grid_all[
            grid_all["monsoon"] == season
        ]
        .set_index(["lat_bin", "lon_bin"])
    )

    d = (
        grid_drogued[
            grid_drogued["monsoon"] == season
        ]
        .set_index(["lat_bin", "lon_bin"])
    )

    comp = a[
        ["mean_ve", "mean_vn", "speed"]
    ].join(
        d[
            ["mean_ve", "mean_vn", "speed"]
        ],
        lsuffix="_all",
        rsuffix="_drogued",
        how="inner"
    )

    # speed difference
    comp["speed_diff"] = (
        comp["speed_all"] -
        comp["speed_drogued"]
    )

    # angle between all-data and drogued-only vectors
    dot = (
        comp["mean_ve_all"] *
        comp["mean_ve_drogued"] +
        comp["mean_vn_all"] *
        comp["mean_vn_drogued"]
    )

    mag_all = np.sqrt(
        comp["mean_ve_all"]**2 +
        comp["mean_vn_all"]**2
    )

    mag_d = np.sqrt(
        comp["mean_ve_drogued"]**2 +
        comp["mean_vn_drogued"]**2
    )

    denom = mag_all * mag_d

    valid = denom > 0

    comp = comp[valid].copy()

    cos_angle = (
        dot[valid] /
        denom[valid]
    )

    cos_angle = np.clip(
        cos_angle,
        -1,
        1
    )

    comp["direction_change_deg"] = np.degrees(
        np.arccos(cos_angle)
    )

    return comp.reset_index()


compare_sw = compare_all_vs_drogued(
    grid_all,
    grid_drogued,
    "SW"
)

compare_ne = compare_all_vs_drogued(
    grid_all,
    grid_drogued,
    "NE"
)

print("\nSW DROGUE SENSITIVITY")
print(
    compare_sw[
        ["speed_diff", "direction_change_deg"]
    ].describe()
)

print("\nNE DROGUE SENSITIVITY")
print(
    compare_ne[
        ["speed_diff", "direction_change_deg"]
    ].describe()
)

for name, comp in [
    ("Southwest", compare_sw),
    ("Northeast", compare_ne)
]:
    print(f"\n{name}")

    for angle in [15, 30, 45, 90]:
        pct = (
            comp["direction_change_deg"] > angle
        ).mean() * 100

        print(
            f">{angle}° direction change: {pct:.1f}%"
        )

# Add drifter counts from the all-data and drogued-only grids

def add_drifter_counts(comp, grid_all, grid_drogued, season):

    a = (
        grid_all[grid_all["monsoon"] == season]
        .set_index(["lat_bin", "lon_bin"])
    )

    d = (
        grid_drogued[grid_drogued["monsoon"] == season]
        .set_index(["lat_bin", "lon_bin"])
    )

    counts = a[["n_drifters"]].join(
        d[["n_drifters"]],
        lsuffix="_all",
        rsuffix="_drogued",
        how="inner"
    )

    comp = comp.set_index(["lat_bin", "lon_bin"])

    comp = comp.join(counts)

    return comp.reset_index()


compare_sw = add_drifter_counts(
    compare_sw,
    grid_all,
    grid_drogued,
    "SW"
)

compare_ne = add_drifter_counts(
    compare_ne,
    grid_all,
    grid_drogued,
    "NE"
)

def sample_group(n):
    if n < 5:
        return "3-4 drifters"
    elif n < 10:
        return "5-9 drifters"
    else:
        return "10+ drifters"


compare_sw["sample_group"] = (
    compare_sw["n_drifters_drogued"].apply(sample_group)
)

compare_ne["sample_group"] = (
    compare_ne["n_drifters_drogued"].apply(sample_group)
)

print("\nSW direction change by sample size")
print(
    compare_sw.groupby("sample_group")["direction_change_deg"]
    .agg(["count", "mean", "median", "max"])
)

print("\nNE direction change by sample size")
print(
    compare_ne.groupby("sample_group")["direction_change_deg"]
    .agg(["count", "mean", "median", "max"])
)

# ============================================================
# DROGUE SENSITIVITY VS SAMPLE SIZE
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------
# 1. Merge all-data and drogued-only grids
# ------------------------------------------------------------

m = allg.merge(
    dr,
    on=["monsoon", "lat_bin", "lon_bin"],
    suffixes=("_all", "_dr")
)

print("\n========================================")
print("DROGUE SENSITIVITY ANALYSIS")
print("========================================")

print(f"\nShared cells: {len(m)}")

print("\nShared cells by monsoon:")
print(m.groupby("monsoon").size())


# ------------------------------------------------------------
# 2. Calculate current speed
# ------------------------------------------------------------

m["speed_all"] = np.sqrt(
    m["ve_all"]**2 +
    m["vn_all"]**2
)

m["speed_dr"] = np.sqrt(
    m["ve_dr"]**2 +
    m["vn_dr"]**2
)


# ------------------------------------------------------------
# 3. Absolute speed difference
# ------------------------------------------------------------
# This is often easier to interpret than % difference.

m["speed_diff_abs"] = (
    m["speed_all"] -
    m["speed_dr"]
)

m["speed_diff_abs_mag"] = np.abs(
    m["speed_diff_abs"]
)


# ------------------------------------------------------------
# 4. Percentage speed difference
# ------------------------------------------------------------
# IMPORTANT:
# If speed_all is extremely close to zero, percentages can become
# enormous and misleading.
#
# Only calculate percentage difference where all-data speed
# is reasonably above zero.

MIN_SPEED_FOR_PERCENT = 0.05

m["pct_diff"] = np.nan

valid_speed = (
    m["speed_all"] >= MIN_SPEED_FOR_PERCENT
)

m.loc[valid_speed, "pct_diff"] = (
    np.abs(
        m.loc[valid_speed, "speed_all"] -
        m.loc[valid_speed, "speed_dr"]
    )
    /
    m.loc[valid_speed, "speed_all"]
    * 100
)


# ------------------------------------------------------------
# 5. Direction of each current vector
# ------------------------------------------------------------

m["angle_all"] = np.degrees(
    np.arctan2(
        m["vn_all"],
        m["ve_all"]
    )
)

m["angle_dr"] = np.degrees(
    np.arctan2(
        m["vn_dr"],
        m["ve_dr"]
    )
)


# Smallest angle between the two directions:
# always between 0 and 180 degrees

m["angle_diff"] = np.abs(
    (
        m["angle_dr"] -
        m["angle_all"] +
        180
    ) % 360 - 180
)


# ------------------------------------------------------------
# 6. Sample-size groups
# ------------------------------------------------------------

def sample_group(n):

    if n < 5:
        return "3-4 drifters"

    elif n < 10:
        return "5-9 drifters"

    else:
        return "10+ drifters"


m["sample_group"] = (
    m["n_dr"]
    .apply(sample_group)
)

# Put groups in logical order
sample_order = [
    "3-4 drifters",
    "5-9 drifters",
    "10+ drifters"
]

m["sample_group"] = pd.Categorical(
    m["sample_group"],
    categories=sample_order,
    ordered=True
)


# ------------------------------------------------------------
# 7. Overall summary by monsoon
# ------------------------------------------------------------

print("\n\nSPEED + DIRECTION DIFFERENCE SUMMARY")

summary = (
    m.groupby("monsoon")[
        [
            "speed_diff_abs",
            "speed_diff_abs_mag",
            "pct_diff",
            "angle_diff"
        ]
    ]
    .describe()
)

print(summary)


# ------------------------------------------------------------
# 8. Correlation with drogued sample size
# ------------------------------------------------------------

print("\n\nRELATIONSHIP WITH NUMBER OF DROGUED DRIFTERS")

for season in ["SW", "NE"]:

    d = m[
        m["monsoon"] == season
    ]

    print(f"\n{season}")

    print(
        "Correlation: n_drifters vs |speed difference| =",
        d["n_dr"].corr(
            d["speed_diff_abs_mag"]
        )
    )

    print(
        "Correlation: n_drifters vs direction difference =",
        d["n_dr"].corr(
            d["angle_diff"]
        )
    )

    # Percentage speed only where valid
    valid = d.dropna(
        subset=["pct_diff"]
    )

    print(
        "Correlation: n_drifters vs % speed difference =",
        valid["n_dr"].corr(
            valid["pct_diff"]
        )
    )


# ------------------------------------------------------------
# 9. Direction sensitivity thresholds
# ------------------------------------------------------------

print("\n\nDIRECTION CHANGE THRESHOLDS")

for season in ["SW", "NE"]:

    d = m[
        m["monsoon"] == season
    ]

    print(f"\n{season}")

    for angle in [
        15,
        30,
        45,
        90
    ]:

        pct = (
            d["angle_diff"] > angle
        ).mean() * 100

        print(
            f">{angle}° direction change: "
            f"{pct:.1f}%"
        )


# ------------------------------------------------------------
# 10. Compare direction change by SAMPLE-SIZE GROUP
# ------------------------------------------------------------

print("\n\nDIRECTION CHANGE BY SAMPLE SIZE")

direction_by_sample = (
    m.groupby(
        ["monsoon", "sample_group"],
        observed=True
    )
    ["angle_diff"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        max="max"
    )
)

print(direction_by_sample)


# ------------------------------------------------------------
# 11. Compare speed change by SAMPLE-SIZE GROUP
# ------------------------------------------------------------

print("\n\nABSOLUTE SPEED CHANGE BY SAMPLE SIZE")

speed_by_sample = (
    m.groupby(
        ["monsoon", "sample_group"],
        observed=True
    )
    ["speed_diff_abs_mag"]
    .agg(
        count="count",
        mean="mean",
        median="median",
        max="max"
    )
)

print(speed_by_sample)


# ------------------------------------------------------------
# 12. Plot SW and NE separately
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13, 10),
    sharex="col",
    constrained_layout=True
)


# ----------------------------
# Southwest
# ----------------------------

sw = m[
    m["monsoon"] == "SW"
]

axes[0, 0].scatter(
    sw["n_dr"],
    sw["speed_diff_abs_mag"],
    alpha=0.5
)

axes[0, 0].set_title(
    "Southwest: Speed Sensitivity"
)

axes[0, 0].set_ylabel(
    "Absolute Speed Difference"
)

axes[0, 0].set_xlabel(
    "Number of Drogued Drifters"
)


axes[0, 1].scatter(
    sw["n_dr"],
    sw["angle_diff"],
    alpha=0.5
)

axes[0, 1].set_title(
    "Southwest: Direction Sensitivity"
)

axes[0, 1].set_ylabel(
    "Direction Difference (degrees)"
)

axes[0, 1].set_xlabel(
    "Number of Drogued Drifters"
)


# ----------------------------
# Northeast
# ----------------------------

ne = m[
    m["monsoon"] == "NE"
]

axes[1, 0].scatter(
    ne["n_dr"],
    ne["speed_diff_abs_mag"],
    alpha=0.5
)

axes[1, 0].set_title(
    "Northeast: Speed Sensitivity"
)

axes[1, 0].set_ylabel(
    "Absolute Speed Difference"
)

axes[1, 0].set_xlabel(
    "Number of Drogued Drifters"
)


axes[1, 1].scatter(
    ne["n_dr"],
    ne["angle_diff"],
    alpha=0.5
)

axes[1, 1].set_title(
    "Northeast: Direction Sensitivity"
)

axes[1, 1].set_ylabel(
    "Direction Difference (degrees)"
)

axes[1, 1].set_xlabel(
    "Number of Drogued Drifters"
)


fig.suptitle(
    "Sensitivity to Drogue Filtering vs Drifter Sample Size",
    fontsize=15
)

plt.show()

x = drifter_cell_drogued["drifter_obs"]
print(x.describe(percentiles=[.1, .25, .5, .75, .9]))
for m in [6, 12, 24, 48, 72]:
    print(f"under {m} obs: {(x < m).mean()*100:.1f}% of visits, "
          f"{x[x < m].sum()/x.sum()*100:.1f}% of observations")

def reversal_share(g):
    sw = g[g.monsoon == "Southwest"].set_index(["lat_bin", "lon_bin"])[["mean_ve", "mean_vn"]]
    ne = g[g.monsoon == "Northeast"].set_index(["lat_bin", "lon_bin"])[["mean_ve", "mean_vn"]]
    b = sw.join(ne, lsuffix="_sw", rsuffix="_ne", how="inner")
    dot = b.mean_ve_sw * b.mean_ve_ne + b.mean_vn_sw * b.mean_vn_ne
    return len(b), (dot < 0).mean()

rows = []
for m in [1, 6, 12, 24, 48, 72]:
    dc = drifter_cell_drogued[drifter_cell_drogued["drifter_obs"] >= m]
    g = (dc.groupby(["monsoon", "lat_bin", "lon_bin"], observed=True)
           .agg(mean_ve=("drifter_ve", "mean"), mean_vn=("drifter_vn", "mean"),
                n_drifters=("drifter_id", "nunique")).reset_index())
    g = g[g.n_drifters >= 3]
    c = g.groupby("monsoon").size()
    n_both, frac = reversal_share(g)
    rows.append(dict(min_obs=m, SW_cells=c.get("Southwest", 0), NE_cells=c.get("Northeast", 0),
                     cells_both=n_both, reversed_pct=round(frac * 100, 1)))
print(pd.DataFrame(rows).set_index("min_obs"))

rng = np.random.default_rng(0)
ses = {}
for m in [1, 12, 24, 48]:
    dc = drifter_cell_drogued[drifter_cell_drogued["drifter_obs"] >= m]
    u = bootstrap_cell_uncertainty(dc)
    ses[m] = u.set_index(["monsoon", "lat_bin", "lon_bin"])["se_ve"]
print(pd.concat(ses, axis=1, join="inner").median())


# ============================================================
# POSTER FIGURES — preliminary results
# ============================================================
# Self-contained: rebuilds everything directly from ds_bob instead of
# reusing variables from earlier sections (build_drifter_weighted_grid,
# grid_drogued, cur, drifter_cell_drogued, etc. are all redefined more
# than once above, so relying on their final state here would silently
# depend on whichever definition happened to run last).
#
# Rule, used throughout: drogued observations only, 1x1 degree cells, a
# cell counts only if at least POSTER_MIN_DRIFTERS distinct drifters each
# contributed at least POSTER_MIN_OBS_PER_DRIFTER hourly observations in
# that cell, and the cell value is a drifter-weighted mean (average
# within each drifter first, then average across qualifying drifters).

POSTER_DIR = "poster_figures"
os.makedirs(POSTER_DIR, exist_ok=True)

POSTER_MIN_OBS_PER_DRIFTER = 12
POSTER_MIN_DRIFTERS = 3
POSTER_GRID_SIZE = 1.0
POSTER_ERR_SST_MAX = 0.5  # degrees C -- see err_sst distribution printed below
POSTER_EICC_LON_RANGE = (80.0, 85.0)
POSTER_N_BOOT = 500
POSTER_SEED = 0

plt.rcParams.update({"font.size": 16})

# --------------------------------------------------
# A. Build the base observation table (SW/NE only, all drogue statuses,
#    so the drogued-share numbers can be computed before filtering)
# --------------------------------------------------

poster_drifter_ids = np.repeat(ds_bob["id"].values, ds_bob["rowsize"].values.astype(int))

poster_all = pd.DataFrame({
    "time": ds_bob["time"].values,
    "lat": ds_bob["lat"].values,
    "lon": ds_bob["lon"].values,
    "ve": ds_bob["ve"].values,
    "vn": ds_bob["vn"].values,
    "sst": ds_bob["sst"].values - 273.15,  # sst is stored in Kelvin (ds_bob["sst"].attrs["units"])
    "err_sst": ds_bob["err_sst"].values,
    "drogue_status": ds_bob["drogue_status"].values,
    "drifter_id": poster_drifter_ids,
})

poster_month = pd.to_datetime(poster_all["time"]).dt.month
poster_all["monsoon"] = np.where(
    poster_month.isin([6, 7, 8, 9]), "Southwest",
    np.where(poster_month.isin([12, 1, 2]), "Northeast", "Inter-monsoon")
)
poster_all = poster_all[poster_all["monsoon"].isin(["Southwest", "Northeast"])].copy()

poster_lat_bins = np.arange(5, 23 + POSTER_GRID_SIZE, POSTER_GRID_SIZE)
poster_lon_bins = np.arange(78, 99 + POSTER_GRID_SIZE, POSTER_GRID_SIZE)
poster_all["lat_bin"] = pd.cut(poster_all["lat"], bins=poster_lat_bins, include_lowest=True)
poster_all["lon_bin"] = pd.cut(poster_all["lon"], bins=poster_lon_bins, include_lowest=True)
poster_all["lat_center"] = poster_all["lat_bin"].apply(lambda x: x.mid).astype(float)
poster_all["lon_center"] = poster_all["lon_bin"].apply(lambda x: x.mid).astype(float)

poster_drogued_share = (
    poster_all.groupby("monsoon")["drogue_status"].mean() * 100
)

poster_obs = poster_all[poster_all["drogue_status"]].copy()  # drogued only, from here on

# --------------------------------------------------
# B. Drifter-weighted velocity grid (for the circulation figure, the
#    reversal statistics and the bootstrap)
# --------------------------------------------------

poster_vel_per_drifter = (
    poster_obs
    .groupby(["monsoon", "lat_bin", "lon_bin", "lat_center", "lon_center", "drifter_id"], observed=True)
    .agg(mean_ve=("ve", "mean"), mean_vn=("vn", "mean"), n_obs=("ve", "count"))
    .reset_index()
)
poster_vel_per_drifter = poster_vel_per_drifter[
    poster_vel_per_drifter["n_obs"] >= POSTER_MIN_OBS_PER_DRIFTER
].copy()


def poster_build_velocity_grid(per_drifter, lon_range=None):
    df = per_drifter
    if lon_range is not None:
        lo, hi = lon_range
        df = df[(df["lon_center"] >= lo) & (df["lon_center"] < hi)]

    cell_grid = (
        df.groupby(["monsoon", "lat_bin", "lon_bin", "lat_center", "lon_center"], observed=True)
        .agg(mean_ve=("mean_ve", "mean"), mean_vn=("mean_vn", "mean"), n_drifters=("drifter_id", "nunique"))
        .reset_index()
    )

    all_cells = df[["monsoon", "lat_bin", "lon_bin", "lat_center", "lon_center"]].drop_duplicates()
    cell_grid = all_cells.merge(
        cell_grid, on=["monsoon", "lat_bin", "lon_bin", "lat_center", "lon_center"], how="left"
    )
    cell_grid["n_drifters"] = cell_grid["n_drifters"].fillna(0).astype(int)

    sufficient = cell_grid[cell_grid["n_drifters"] >= POSTER_MIN_DRIFTERS].copy()
    sufficient["speed"] = np.sqrt(sufficient["mean_ve"]**2 + sufficient["mean_vn"]**2)
    sufficient = sufficient[sufficient["speed"] > 0].copy()
    sufficient["u_norm"] = sufficient["mean_ve"] / sufficient["speed"]
    sufficient["v_norm"] = sufficient["mean_vn"] / sufficient["speed"]

    insufficient = cell_grid[cell_grid["n_drifters"] < POSTER_MIN_DRIFTERS].copy()
    return sufficient, insufficient


poster_vel_sufficient, poster_vel_insufficient = poster_build_velocity_grid(poster_vel_per_drifter)

poster_cells_kept = poster_vel_sufficient.groupby("monsoon").size()

# --------------------------------------------------
# C. Reversal share (>90 degree direction change) + bootstrap CI,
#    whole bay and the approximate EICC band
# --------------------------------------------------


def poster_reversal_point_estimate(per_drifter, lon_range=None):
    df = per_drifter
    if lon_range is not None:
        lo, hi = lon_range
        df = df[(df["lon_center"] >= lo) & (df["lon_center"] < hi)]

    g = (
        df.groupby(["monsoon", "lat_bin", "lon_bin"], observed=True)
        .agg(mean_ve=("mean_ve", "mean"), mean_vn=("mean_vn", "mean"), n_drifters=("drifter_id", "nunique"))
        .reset_index()
    )
    g = g[g["n_drifters"] >= POSTER_MIN_DRIFTERS]

    sw = g[g["monsoon"] == "Southwest"].set_index(["lat_bin", "lon_bin"])[["mean_ve", "mean_vn"]]
    ne = g[g["monsoon"] == "Northeast"].set_index(["lat_bin", "lon_bin"])[["mean_ve", "mean_vn"]]
    both = sw.join(ne, lsuffix="_sw", rsuffix="_ne", how="inner")
    if len(both) == 0:
        return 0, np.nan

    dot = both["mean_ve_sw"] * both["mean_ve_ne"] + both["mean_vn_sw"] * both["mean_vn_ne"]
    return len(both), (dot < 0).mean() * 100


def poster_bootstrap_reversal(per_drifter, lon_range=None, n_boot=POSTER_N_BOOT, seed=POSTER_SEED):
    df = per_drifter
    if lon_range is not None:
        lo, hi = lon_range
        df = df[(df["lon_center"] >= lo) & (df["lon_center"] < hi)]

    unique_drifters = df["drifter_id"].unique()
    n_drifters = len(unique_drifters)
    rng = np.random.default_rng(seed)

    results = []
    for _ in range(n_boot):
        draws = rng.choice(unique_drifters, size=n_drifters, replace=True)
        draw_counts = pd.Series(draws).value_counts()

        # Resampled "instances": duplicated draws of the same physical
        # drifter are weighted copies, not collapsed, so a drifter drawn
        # twice counts twice toward both the cell mean and n_drifters --
        # standard whole-unit bootstrap behaviour.
        tmp = df.merge(draw_counts.rename("w"), left_on="drifter_id", right_index=True, how="inner")
        tmp["w_ve"] = tmp["mean_ve"] * tmp["w"]
        tmp["w_vn"] = tmp["mean_vn"] * tmp["w"]

        g = (
            tmp.groupby(["monsoon", "lat_bin", "lon_bin"], observed=True)
            .agg(sum_ve=("w_ve", "sum"), sum_vn=("w_vn", "sum"), sum_w=("w", "sum"))
            .reset_index()
        )
        g["mean_ve"] = g["sum_ve"] / g["sum_w"]
        g["mean_vn"] = g["sum_vn"] / g["sum_w"]
        g = g[g["sum_w"] >= POSTER_MIN_DRIFTERS]

        sw = g[g["monsoon"] == "Southwest"].set_index(["lat_bin", "lon_bin"])[["mean_ve", "mean_vn"]]
        ne = g[g["monsoon"] == "Northeast"].set_index(["lat_bin", "lon_bin"])[["mean_ve", "mean_vn"]]
        both = sw.join(ne, lsuffix="_sw", rsuffix="_ne", how="inner")
        if len(both) == 0:
            continue

        dot = both["mean_ve_sw"] * both["mean_ve_ne"] + both["mean_vn_sw"] * both["mean_vn_ne"]
        results.append((dot < 0).mean())

    results = np.array(results)
    return np.median(results) * 100, np.percentile(results, 2.5) * 100, np.percentile(results, 97.5) * 100


poster_n_both, poster_reversed_pct = poster_reversal_point_estimate(poster_vel_per_drifter)
poster_rev_med, poster_rev_lo, poster_rev_hi = poster_bootstrap_reversal(poster_vel_per_drifter)

poster_n_both_eicc, poster_reversed_pct_eicc = poster_reversal_point_estimate(
    poster_vel_per_drifter, lon_range=POSTER_EICC_LON_RANGE
)
poster_rev_med_eicc, poster_rev_lo_eicc, poster_rev_hi_eicc = poster_bootstrap_reversal(
    poster_vel_per_drifter, lon_range=POSTER_EICC_LON_RANGE
)

# --------------------------------------------------
# D. Circulation figure
# --------------------------------------------------

poster_fig1, poster_axes1 = plt.subplots(1, 2, figsize=(16, 8))

poster_vel_norm = Normalize(vmin=0, vmax=0.9)
poster_vel_cmap = plt.get_cmap("YlGnBu")

for ax, season in zip(poster_axes1, ["Southwest", "Northeast"]):
    season_data = poster_vel_sufficient[poster_vel_sufficient["monsoon"] == season]
    season_insufficient = poster_vel_insufficient[poster_vel_insufficient["monsoon"] == season]

    for _, cell in season_insufficient.iterrows():
        ax.add_patch(Rectangle(
            (cell["lon_center"] - POSTER_GRID_SIZE / 2, cell["lat_center"] - POSTER_GRID_SIZE / 2),
            POSTER_GRID_SIZE, POSTER_GRID_SIZE,
            facecolor="lightgrey", edgecolor="none", hatch="//", alpha=0.6, zorder=0
        ))

    for _, cell in season_data.iterrows():
        ax.add_patch(Rectangle(
            (cell["lon_center"] - POSTER_GRID_SIZE / 2, cell["lat_center"] - POSTER_GRID_SIZE / 2),
            POSTER_GRID_SIZE, POSTER_GRID_SIZE,
            facecolor=poster_vel_cmap(poster_vel_norm(cell["speed"])), edgecolor="none", zorder=0
        ))

    ax.quiver(
        season_data["lon_center"], season_data["lat_center"],
        season_data["u_norm"], season_data["v_norm"],
        color="black", angles="xy", scale=25, width=0.003
    )

    ax.set_xlim(78, 99)
    ax.set_ylim(5, 23)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{season} Monsoon")
    ax.set_facecolor("white")
    ax.grid(alpha=0.2)

poster_insufficient_patch = Rectangle(
    (0, 0), 1, 1, facecolor="lightgrey", edgecolor="none", hatch="//", alpha=0.6,
    label=f"Insufficient data (<{POSTER_MIN_DRIFTERS} drifters)"
)
poster_axes1[0].legend(handles=[poster_insufficient_patch], loc="upper left")

poster_cbar1 = poster_fig1.colorbar(
    plt.cm.ScalarMappable(norm=poster_vel_norm, cmap=poster_vel_cmap),
    ax=poster_axes1, location="right", shrink=0.85, pad=0.03
)
poster_cbar1.set_label("Mean current speed (m/s)")

poster_fig1.patch.set_facecolor("white")
poster_fig1.savefig(
    os.path.join(POSTER_DIR, "circulation_SW_NE.png"),
    dpi=300, bbox_inches="tight", facecolor="white"
)
plt.close(poster_fig1)

# --------------------------------------------------
# E. SST distribution + threshold, then SST figure
# --------------------------------------------------

print("\nERR_SST DISTRIBUTION (for poster SST threshold)")
poster_err_sst_all = pd.Series(ds_bob["err_sst"].values).dropna()
print(poster_err_sst_all.describe(percentiles=[.5, .9, .95, .975, .99, .995, .999]))
print(f"max: {poster_err_sst_all.max():.3f}")
print(
    f"Dropping err_sst > {POSTER_ERR_SST_MAX}: removes "
    f"{(poster_err_sst_all > POSTER_ERR_SST_MAX).mean() * 100:.2f}% of non-null err_sst observations"
)

poster_sst_obs = poster_obs.dropna(subset=["sst", "err_sst"])
poster_sst_obs = poster_sst_obs[poster_sst_obs["err_sst"] <= POSTER_ERR_SST_MAX].copy()

poster_sst_per_drifter = (
    poster_sst_obs
    .groupby(["monsoon", "lat_bin", "lon_bin", "lat_center", "lon_center", "drifter_id"], observed=True)
    .agg(mean_sst=("sst", "mean"), n_obs=("sst", "count"))
    .reset_index()
)
poster_sst_per_drifter = poster_sst_per_drifter[
    poster_sst_per_drifter["n_obs"] >= POSTER_MIN_OBS_PER_DRIFTER
].copy()

poster_sst_cell_grid = (
    poster_sst_per_drifter
    .groupby(["monsoon", "lat_bin", "lon_bin", "lat_center", "lon_center"], observed=True)
    .agg(mean_sst=("mean_sst", "mean"), n_drifters=("drifter_id", "nunique"))
    .reset_index()
)

poster_sst_all_cells = poster_sst_obs[
    ["monsoon", "lat_bin", "lon_bin", "lat_center", "lon_center"]
].drop_duplicates()
poster_sst_cell_grid = poster_sst_all_cells.merge(
    poster_sst_cell_grid, on=["monsoon", "lat_bin", "lon_bin", "lat_center", "lon_center"], how="left"
)
poster_sst_cell_grid["n_drifters"] = poster_sst_cell_grid["n_drifters"].fillna(0).astype(int)

poster_sst_sufficient = poster_sst_cell_grid[poster_sst_cell_grid["n_drifters"] >= POSTER_MIN_DRIFTERS].copy()
poster_sst_insufficient = poster_sst_cell_grid[poster_sst_cell_grid["n_drifters"] < POSTER_MIN_DRIFTERS].copy()

poster_fig2, poster_axes2 = plt.subplots(1, 2, figsize=(16, 8))

poster_sst_norm = Normalize(
    vmin=poster_sst_sufficient["mean_sst"].min(), vmax=poster_sst_sufficient["mean_sst"].max()
)
poster_sst_cmap = plt.get_cmap("YlOrRd")

for ax, season in zip(poster_axes2, ["Southwest", "Northeast"]):
    season_data = poster_sst_sufficient[poster_sst_sufficient["monsoon"] == season]
    season_insufficient = poster_sst_insufficient[poster_sst_insufficient["monsoon"] == season]

    for _, cell in season_insufficient.iterrows():
        ax.add_patch(Rectangle(
            (cell["lon_center"] - POSTER_GRID_SIZE / 2, cell["lat_center"] - POSTER_GRID_SIZE / 2),
            POSTER_GRID_SIZE, POSTER_GRID_SIZE,
            facecolor="lightgrey", edgecolor="none", hatch="//", alpha=0.6, zorder=0
        ))

    for _, cell in season_data.iterrows():
        ax.add_patch(Rectangle(
            (cell["lon_center"] - POSTER_GRID_SIZE / 2, cell["lat_center"] - POSTER_GRID_SIZE / 2),
            POSTER_GRID_SIZE, POSTER_GRID_SIZE,
            facecolor=poster_sst_cmap(poster_sst_norm(cell["mean_sst"])), edgecolor="none", zorder=0
        ))

    ax.set_xlim(78, 99)
    ax.set_ylim(5, 23)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"{season} Monsoon")
    ax.set_facecolor("white")
    ax.grid(alpha=0.2)

poster_axes2[0].legend(handles=[poster_insufficient_patch], loc="upper left")

poster_cbar2 = poster_fig2.colorbar(
    plt.cm.ScalarMappable(norm=poster_sst_norm, cmap=poster_sst_cmap),
    ax=poster_axes2, location="right", shrink=0.85, pad=0.03
)
poster_cbar2.set_label("Mean sea surface temperature (°C)")

poster_fig2.patch.set_facecolor("white")
poster_fig2.savefig(
    os.path.join(POSTER_DIR, "sst_SW_NE.png"),
    dpi=300, bbox_inches="tight", facecolor="white"
)
plt.close(poster_fig2)

poster_sst_cells_kept = poster_sst_sufficient.groupby("monsoon").size()

# --------------------------------------------------
# F. Numbers file
# --------------------------------------------------

poster_lines = []
poster_lines.append("POSTER PRELIMINARY RESULTS -- NUMBERS")
poster_lines.append(f"Rule: drogued only, 1x1 deg cells, >= {POSTER_MIN_DRIFTERS} drifters each with "
                     f">= {POSTER_MIN_OBS_PER_DRIFTER} hourly obs, drifter-weighted mean")
poster_lines.append("")
poster_lines.append(f"Bay of Bengal observations (SW+NE months): {len(poster_all)}")
poster_lines.append(f"Distinct drifters (SW+NE months): {poster_all['drifter_id'].nunique()}")
poster_lines.append("")
poster_lines.append("Circulation map -- cells kept per season:")
poster_lines.append(f"  Southwest: {poster_cells_kept.get('Southwest', 0)}")
poster_lines.append(f"  Northeast: {poster_cells_kept.get('Northeast', 0)}")
poster_lines.append("")
poster_lines.append("Drogued share of observations, per season:")
poster_lines.append(f"  Southwest: {poster_drogued_share.get('Southwest', float('nan')):.1f}%")
poster_lines.append(f"  Northeast: {poster_drogued_share.get('Northeast', float('nan')):.1f}%")
poster_lines.append("")
poster_lines.append(f"Cells with data in both seasons: {poster_n_both}")
poster_lines.append("")
poster_lines.append("SW vs NE direction reversal (>90 degrees), whole bay:")
poster_lines.append(f"  point estimate: {poster_reversed_pct:.1f}% ({poster_n_both} cells)")
poster_lines.append(
    f"  bootstrap ({POSTER_N_BOOT} resamples, whole drifters): median {poster_rev_med:.1f}%, "
    f"95% CI ({poster_rev_lo:.1f}%, {poster_rev_hi:.1f}%)"
)
poster_lines.append("")
poster_lines.append(
    f"SW vs NE direction reversal (>90 degrees), {POSTER_EICC_LON_RANGE[0]:.0f}-"
    f"{POSTER_EICC_LON_RANGE[1]:.0f}E band (approx. EICC):"
)
poster_lines.append(f"  point estimate: {poster_reversed_pct_eicc:.1f}% ({poster_n_both_eicc} cells)")
poster_lines.append(
    f"  bootstrap ({POSTER_N_BOOT} resamples, whole drifters): median {poster_rev_med_eicc:.1f}%, "
    f"95% CI ({poster_rev_lo_eicc:.1f}%, {poster_rev_hi_eicc:.1f}%)"
)
poster_lines.append("")
poster_lines.append("SST map -- cells kept per season "
                     f"(err_sst <= {POSTER_ERR_SST_MAX}, same drifter/obs rule):")
poster_lines.append(f"  Southwest: {poster_sst_cells_kept.get('Southwest', 0)}")
poster_lines.append(f"  Northeast: {poster_sst_cells_kept.get('Northeast', 0)}")

poster_numbers_text = "\n".join(poster_lines)
print("\n" + poster_numbers_text)

with open(os.path.join(POSTER_DIR, "poster_numbers.txt"), "w") as f:
    f.write(poster_numbers_text + "\n")

plt.rcParams.update({"font.size": plt.rcParamsDefault["font.size"]})