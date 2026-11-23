"""EDA figures and numbers (EDA section, Figures E1-E6 and Tables E1-E4).

Figures: drifter tracks, records per year, drogued drifters per month, speed
distribution, days of data per drifter. Numbers (records, removals, drogued
share, 2015+2019 share, speeds, ...) go to eda_numbers.json.
Figures E7-E11 (observation density, entry points, drogue by monsoon, naive and
drifter-weighted circulation) come from project3001.py in the main repository.
"""
import os, json
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import xarray as xr
import reversal_figures as rf

from common import DATA, OUT
NAVY, BLUE, ORANGE, GREY = "#00325a", "#2b6cb0", "#e07b39", "#9aa5b1"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17, "axes.labelsize": 19,
                     "xtick.labelsize": 16, "ytick.labelsize": 16, "legend.fontsize": 16,
                     "axes.spines.top": False, "axes.spines.right": False})
def title(ax, t): ax.set_title(t, fontsize=24, fontweight="bold", color=NAVY, pad=12)
def save(fig, n): fig.savefig(OUT + n, dpi=150, facecolor="white"); plt.close(fig); print("saved", n)
def coast(ax):
    for line in [rf.INDIA_COAST, rf.LANKA_COAST]:
        a = np.asarray(line); ax.plot(a[:, 0], a[:, 1], color="#333", lw=1.4)
def mapaxes(ax):
    ax.set_xlim(78, 99); ax.set_ylim(5, 23); ax.set_aspect("equal")
    ax.set_xticks([80, 85, 90, 95]); ax.set_yticks([5, 10, 15, 20])
    ax.set_xticklabels(["80°E", "85°E", "90°E", "95°E"]); ax.set_yticklabels(["5°N", "10°N", "15°N", "20°N"])

ds = xr.open_dataset(DATA)          # raw records, before cleaning
raw = pd.DataFrame({"time": pd.to_datetime(ds["time"].values), "lat": ds["lat"].values, "lon": ds["lon"].values,
                    "ve": ds["ve"].values, "vn": ds["vn"].values, "drogue": ds["drogue_status"].values.astype(bool),
                    "drifter_id": np.repeat(ds["id"].values, ds["rowsize"].values.astype(int))})
T = {}
T["raw_records"] = len(raw); T["raw_drifters"] = int(raw.drifter_id.nunique())
T["missing"] = int(raw[["time", "lat", "lon", "ve", "vn"]].isna().any(axis=1).sum())
T["duplicates"] = int(raw.duplicated(["drifter_id", "time"]).sum())
T["too_fast"] = int((np.hypot(raw.ve, raw.vn) > rf.MAX_SPEED).sum())
df = rf.clean_observations(raw); df["speed"] = np.hypot(df.ve, df.vn); df["year"] = df.time.dt.year
T["clean_records"] = len(df); T["clean_drifters"] = int(df.drifter_id.nunique())
dr = df[df.drogue]
T["drogued_records"] = len(dr); T["drogued_drifters"] = int(dr.drifter_id.nunique())
T["drogued_pct"] = round(len(dr) / len(df) * 100, 1)
T["start"], T["end"] = str(df.time.min().date()), str(df.time.max().date())

# by season
seas = []
for s in ["Southwest", "Northeast", "Inter"]:
    a, b = df[df.season == s], dr[dr.season == s]
    seas.append([s, len(a), int(a.drifter_id.nunique()), round(len(b) / len(a) * 100), len(b), int(b.drifter_id.nunique())])
T["season"] = seas
# by year
y_all = df.groupby("year").size(); y_dr = dr.groupby("year").size().reindex(y_all.index, fill_value=0)
T["share_2015_2019"] = round((y_all.get(2015, 0) + y_all.get(2019, 0)) / len(df) * 100, 1)
T["share_2015_2019_drogued"] = round((y_dr.get(2015, 0) + y_dr.get(2019, 0)) / len(dr) * 100, 1)
# records per drifter (drogued)
rpd = dr.groupby("drifter_id").size() / 24
T["days_per_drifter_median"] = round(float(rpd.median()), 1); T["days_per_drifter_max"] = round(float(rpd.max()))
top10 = rpd.sort_values(ascending=False); T["top10pct_share"] = round(top10.iloc[:max(1, len(top10) // 10)].sum() / top10.sum() * 100, 1)
# speeds
T["speed_med_drogued"] = round(float(dr.speed.median()), 2)
T["speed_med_undrogued"] = round(float(df[~df.drogue].speed.median()), 2)
T["speed_p99_drogued"] = round(float(dr.speed.quantile(.99)), 2)
# monthly drogued drifters
T["drifters_by_month"] = [int(dr[dr.month == m].drifter_id.nunique()) for m in range(1, 13)]
json.dump(T, open(OUT + "eda_numbers.json", "w"), indent=1); print(json.dumps(T, indent=1))

# ---- 1. tracks
fig, ax = plt.subplots(figsize=(12, 10.5)); fig.subplots_adjust(left=.12, right=.97, top=.92, bottom=.1)
d2 = df.sort_values(["drifter_id", "time"])
for (did, dg), g in d2.groupby(["drifter_id", "drogue"]):
    ax.plot(g.lon.values[::6], g.lat.values[::6], lw=.5, color=BLUE if dg else ORANGE, alpha=.45)
coast(ax); mapaxes(ax); title(ax, "Drifter tracks")
ax.legend(handles=[Line2D([], [], color=BLUE, lw=3, label="Drogued"), Line2D([], [], color=ORANGE, lw=3, label="Undrogued")],
          loc="lower right", frameon=True)
save(fig, "eda_tracks.png")

# ---- 2. records per year
fig, ax = plt.subplots(figsize=(15, 7.5)); fig.subplots_adjust(left=.1, right=.98, top=.88, bottom=.15)
x = y_all.index.values
ax.bar(x, y_dr.values / 1000, color=BLUE, label="Drogued")
ax.bar(x, (y_all - y_dr).values / 1000, bottom=y_dr.values / 1000, color=ORANGE, label="Undrogued")
for yr in (2015, 2019):
    ax.text(yr, y_all[yr] / 1000 + 3, str(yr), ha="center", fontsize=16, fontweight="bold", color=NAVY)
ax.set_ylabel("Hourly records (thousands)"); ax.set_xlabel("Year"); ax.set_xticks(range(2000, 2023, 2))
ax.legend(frameon=False, loc="upper left"); title(ax, "Records per year")
save(fig, "eda_years.png")

# ---- 3. drogued drifters per month
fig, ax = plt.subplots(figsize=(13, 6.8)); fig.subplots_adjust(left=.1, right=.98, top=.86, bottom=.14)
cols = ["#2b6cb0" if m in (12, 1, 2) else "#e07b39" if m in (6, 7, 8, 9) else GREY for m in range(1, 13)]
ax.bar(range(1, 13), T["drifters_by_month"], color=cols)
for m, v in zip(range(1, 13), T["drifters_by_month"]): ax.text(m, v + 2, str(v), ha="center", fontsize=15)
ax.set_xticks(range(1, 13)); ax.set_xticklabels(rf.MONTH_NAMES); ax.set_ylabel("Drogued drifters")
ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#e07b39"), plt.Rectangle((0, 0), 1, 1, color="#2b6cb0"),
                   plt.Rectangle((0, 0), 1, 1, color=GREY)], labels=["Southwest monsoon", "Northeast monsoon", "Between monsoons"],
          frameon=False, loc="upper left", ncol=3)
ax.set_ylim(0, max(T["drifters_by_month"]) * 1.25); title(ax, "Drifters per month")
save(fig, "eda_months.png")

# ---- 4. speed distribution
fig, ax = plt.subplots(figsize=(13, 7)); fig.subplots_adjust(left=.1, right=.98, top=.88, bottom=.14)
bins = np.linspace(0, 1.6, 65)
ax.hist(dr.speed, bins=bins, density=True, histtype="step", lw=3, color=BLUE, label=f"Drogued (median {T['speed_med_drogued']} m/s)")
ax.hist(df[~df.drogue].speed, bins=bins, density=True, histtype="step", lw=3, color=ORANGE, label=f"Undrogued (median {T['speed_med_undrogued']} m/s)")
ax.axvline(.1, color="grey", ls=":", lw=2); ax.text(.12, .25, "0.1 m/s cut-off", fontsize=15, color="grey")
ax.set_xlabel("Current speed (m/s)"); ax.set_ylabel("Density"); ax.legend(frameon=False); title(ax, "Speed distribution")
save(fig, "eda_speed.png")

# ---- 5. days per drifter
fig, ax = plt.subplots(figsize=(13, 7)); fig.subplots_adjust(left=.1, right=.98, top=.88, bottom=.14)
ax.hist(rpd, bins=40, color=BLUE)
ax.axvline(rpd.median(), color=NAVY, ls="--", lw=2.5); ax.text(rpd.median() + 8, ax.get_ylim()[1] * .9, f"median {rpd.median():.0f} days", fontsize=16, color=NAVY)
ax.set_xlabel("Days of drogued data per drifter"); ax.set_ylabel("Number of drifters"); title(ax, "Time per drifter")
save(fig, "eda_days_per_drifter.png")
