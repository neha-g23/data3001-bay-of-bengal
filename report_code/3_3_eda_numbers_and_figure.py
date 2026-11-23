"""Section 3.2-3.3 numbers and Figure E1 (exploratory data analysis).

Numbers (records, removals, drogued share, 2015+2019 share, days per drifter,
speeds, drifters per month, ...) go to eda_numbers.json.
Figure E1 (figure_e1_eda.png): (a) hourly records per year, drogued and undrogued;
(b) drogued drifters per month; (c) speed distribution with the 0.1 m/s cut-off;
(d) days of drogued data per drifter.
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

# ---- Figure E1: four panels
fig, axs = plt.subplots(2, 2, figsize=(24, 13)); fig.subplots_adjust(left=.05, right=.98, top=.93, bottom=.07, hspace=.38, wspace=.15)
(a, b), (c, d) = axs
x = y_all.index.values
a.bar(x, y_dr.values / 1000, color=BLUE, label="Drogued")
a.bar(x, (y_all - y_dr).values / 1000, bottom=y_dr.values / 1000, color=ORANGE, label="Undrogued")
for yr in (2015, 2019):
    a.text(yr, y_all[yr] / 1000 + 3, str(yr), ha="center", fontsize=16, fontweight="bold", color=NAVY)
a.set_ylabel("Hourly records (thousands)"); a.set_xlabel("Year"); a.set_xticks(range(2000, 2023, 2))
a.legend(frameon=False, loc="upper left"); title(a, "Records per year")
cols = ["#2b6cb0" if m in (12, 1, 2) else "#e07b39" if m in (6, 7, 8, 9) else GREY for m in range(1, 13)]
b.bar(range(1, 13), T["drifters_by_month"], color=cols)
for m, v in zip(range(1, 13), T["drifters_by_month"]): b.text(m, v + 2, str(v), ha="center", fontsize=15)
b.set_xticks(range(1, 13)); b.set_xticklabels(rf.MONTH_NAMES); b.set_ylabel("Drogued drifters")
b.legend(handles=[plt.Rectangle((0, 0), 1, 1, color="#e07b39"), plt.Rectangle((0, 0), 1, 1, color="#2b6cb0"),
                  plt.Rectangle((0, 0), 1, 1, color=GREY)], labels=["Southwest monsoon", "Northeast monsoon", "Between monsoons"],
         frameon=False, loc="upper left", ncol=3)
b.set_ylim(0, max(T["drifters_by_month"]) * 1.25); title(b, "Drifters per month")
bins = np.linspace(0, 1.6, 65)
c.hist(dr.speed, bins=bins, density=True, histtype="step", lw=3, color=BLUE, label=f"Drogued (median {T['speed_med_drogued']} m/s)")
c.hist(df[~df.drogue].speed, bins=bins, density=True, histtype="step", lw=3, color=ORANGE, label=f"Undrogued (median {T['speed_med_undrogued']} m/s)")
c.axvline(.1, color="grey", ls=":", lw=2); c.text(.12, .25, "0.1 m/s cut-off", fontsize=15, color="grey")
c.set_xlabel("Current speed (m/s)"); c.set_ylabel("Density"); c.legend(frameon=False); title(c, "Speed distribution")
d.hist(rpd, bins=40, color=BLUE)
d.axvline(rpd.median(), color=NAVY, ls="--", lw=2.5); d.text(rpd.median() + 8, d.get_ylim()[1] * .9, f"median {rpd.median():.0f} days", fontsize=16, color=NAVY)
d.set_xlabel("Days of drogued data per drifter"); d.set_ylabel("Number of drifters"); title(d, "Time per drifter")
for ax, lab in zip([a, b, c, d], "abcd"):
    ax.text(-.08, 1.12, f"({lab})", transform=ax.transAxes, fontsize=30, fontweight="bold", color=NAVY)
save(fig, "figure_e1_eda.png")
