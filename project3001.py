import os
import clouddrift as cd
from clouddrift.ragged import subset
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np



cache_path = "bob_subset.nc"

if os.path.exists(cache_path):
    # Already computed before — just load it
    print("Loading cached subset...")
    import xarray as xr
    ds_bob = xr.open_dataset(cache_path)
else:
    # First time — compute it and save
    print("Computing subset (this will take a few mins)...")
    ds = cd.datasets.gdp1h()
    ds_bob = subset(
        ds, 
        {"lat": (5.0, 23.0), "lon": (78.0, 99.0)}, 
        row_dim_name="traj"
    )
    ds_bob.load()  # pull it fully into memory before saving
    ds_bob.to_netcdf(cache_path)
    print("Saved subset to disk.")

# num of observations: 866511
print(f"Bay of Bengal observations: {ds_bob.sizes['obs']}")

# num of diff drifters: 388
n_drifters = ds_bob.sizes['traj']
print(f"Number of distinct drifters: {n_drifters}")

# Make a density map / count observations per grid cell. also shows spatial coverage
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

# decent seasonal coverage
# Pull time into a dataframe
print("SEASONAL COVERAGE\n")
df_time = ds_bob[["time"]].to_dataframe().reset_index()

# Extract month from each observation's timestamp
df_time["month"] = pd.to_datetime(df_time["time"]).dt.month

# Count observations per month
monthly_counts = df_time["month"].value_counts().sort_index()
print(monthly_counts)

# Plot it
plt.figure(figsize=(10, 5))
monthly_counts.plot(kind="bar", color="steelblue")
plt.xlabel("Month")
plt.ylabel("Number of Observations")
plt.title("Seasonal Coverage: Observations per Month — Bay of Bengal")
plt.xticks(rotation=0)
plt.show()



# Core measurements + uncertainty + quality flags + drogue status
cols = [
    "time", "lat", "lon",
    "ve", "vn", "err_ve", "err_vn",
    "sst", "sst1", "sst2", "err_sst", "err_sst1", "err_sst2",
    "flg_sst", "flg_sst1", "flg_sst2",
    "drogue_status", "gap"
]

df = ds_bob[cols].to_dataframe()

# Missing value counts
missing_counts = df.isna().sum()
print("Missing value counts:\n", missing_counts)

missing_pct = (df.isna().sum() / len(df)) * 100
print("\nMissing value %:\n", missing_pct)

# Summary stats — check for suspicious fill values (e.g. -999) or unrealistic ranges
print("\nSummary stats:\n", df.describe())

# Quality flag breakdowns
print("\nflg_sst counts:\n", df["flg_sst"].value_counts())
print("\ndrogue_status counts:\n", df["drogue_status"].value_counts())

# Check for outliers 

#### moonsoon obs

print("\n\nMOONSOON")
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
plt.show()

#########


#obs per year
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
plt.show()

# drouged vs undrouged
print(df["drogue_status"].value_counts(dropna=False))

#calc percentGES
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

# Rename columns so legend is readable
drogue_by_monsoon = drogue_by_monsoon.rename(
    columns={
        False: "Non-drogued",
        True: "Drogued"
    }
)

# Shorter labels for the figure
drogue_by_monsoon.index = [
    "Inter-monsoon",
    "Northeast",
    "Southwest"
]

# Plot
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
# ##############################

#TRajectorid

print("TRAJECTORY MAP\n")

# We need the ragged trajectory sizes
rowsize = ds_bob["rowsize"].values

# Observation coordinates
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
plt.show()


##naive 

print("\nNAIVE MONSOON CIRCULATION MAPS\n")

# --------------------------------------------------
# 1. Make a dataframe with the variables we need
# --------------------------------------------------

currents = ds_bob[
    ["time", "lat", "lon", "ve", "vn"]
].to_dataframe().reset_index()

# Remove observations with missing current/location data
currents = currents.dropna(
    subset=["time", "lat", "lon", "ve", "vn"]
)

# --------------------------------------------------
# 2. Label each observation by monsoon
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
# 3. Create a 1° x 1° spatial grid
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
# 4. Calculate average current in each grid cell
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

# Centre coordinates of each grid cell
grid["lat_center"] = grid["lat_bin"].apply(
    lambda x: x.mid
).astype(float)

grid["lon_center"] = grid["lon_bin"].apply(
    lambda x: x.mid
).astype(float)

# --------------------------------------------------
# 5. Only show cells with enough observations
# --------------------------------------------------

min_obs = 20

grid = grid[grid["n_obs"] >= min_obs]

print("\nCells remaining after minimum observation threshold:")
print(grid.groupby("monsoon").size())

# --------------------------------------------------
# 6. Plot Southwest vs Northeast
# --------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(15, 7),
    sharex=True,
    sharey=True,
    constrained_layout=True
)

vmax = 0

# Find common max speed across both monsoons
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
        cmap="turbo",   # changed from viridis
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

# Add one shared colorbar
cbar = fig.colorbar(
    q,
    ax=axes,
    location="right",
    shrink=0.85,
    pad=0.03
)
cbar.set_label("Mean Current Speed")

fig.suptitle("Mean Surface Current Direction and Speed: Bay of Bengal", fontsize=16)

plt.show()


print("\nOBSERVATIONS AND DRIFTERS PER MONSOON\n")

# ---------------------------------------
# 1. Give every observation its drifter ID
# ---------------------------------------

drifter_ids = np.repeat(
    ds_bob["id"].values,
    ds_bob["rowsize"].values.astype(int)
)

# Check it matches number of observations
print("Observation rows:", ds_bob.sizes["obs"])
print("Drifter ID rows:", len(drifter_ids))


# ---------------------------------------
# 2. Create dataframe
# ---------------------------------------

season_df = pd.DataFrame({
    "time": ds_bob["time"].values,
    "drifter_id": drifter_ids
})

season_df["month"] = pd.to_datetime(
    season_df["time"]
).dt.month


# ---------------------------------------
# 3. Label monsoon period
# ---------------------------------------

def get_monsoon(month):
    if month in [6, 7, 8, 9]:
        return "Southwest Monsoon"
    elif month in [12, 1, 2]:
        return "Northeast Monsoon"
    else:
        return "Inter-monsoon"

season_df["monsoon"] = season_df["month"].apply(get_monsoon)


# ---------------------------------------
# 4. Count observations AND unique drifters
# ---------------------------------------

monsoon_summary = (
    season_df
    .groupby("monsoon")
    .agg(
        observations=("time", "count"),
        unique_drifters=("drifter_id", "nunique")
    )
)

print(monsoon_summary)