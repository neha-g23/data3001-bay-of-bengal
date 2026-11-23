"""Headline reversal share with 95% range, and how consistently each location reverses.

Report: 82% of the measured Bay reverses (95% range 72-88%); 64% of the area
reverses in at least 95% of redraws ("consistently reverses").
Saves agreement_pct.npy (used by 03_region_table.py), headline_bootstrap.csv and
agreement_map.png (Q3 figure: share of redraws in which each location reverses).
Uses 500 redraws by default (set NB to change).

Sensitivity tests (bandwidth, minimum speed, cut-off, leaving out 2015/2019,
all drifters) are in reversal_uncertainty.py:
    python reversal_uncertainty.py --data bob_subset.nc --out report_outputs
"""
import os
import numpy as np, pandas as pd
import reversal_figures as rf, reversal_uncertainty as ru
from common import load, OUT, SEED, redraw

NB = int(os.environ.get("NB", 500))
df = load(); grid = rf.Grid(0.2, df)
W, nd = ru.build(df, "drogued", grid)
full = {s: W[s].field(np.ones(nd)) for s in W}
base, fast0, rev0 = ru.summarise(full["Southwest"], full["Northeast"], grid)

rng = np.random.default_rng(SEED)
rows, agree, counted = [], np.zeros(grid.LAT.shape), np.zeros(grid.LAT.shape)
for _ in range(NB):
    f = {s: W[s].field(redraw(rng, nd)) for s in W}
    res, fast, rev = ru.summarise(f["Southwest"], f["Northeast"], grid)
    rows.append(res); agree += fast & rev; counted += fast
B = pd.DataFrame(rows)
T = pd.DataFrame({"estimate": pd.Series(base), "lower 95%": B.quantile(.025), "upper 95%": B.quantile(.975)}).round(1)
T.to_csv(OUT + "headline_bootstrap.csv")
with np.errstate(invalid="ignore", divide="ignore"):
    pct = np.where(fast0, agree / np.maximum(counted, 1) * 100, np.nan)
np.save(OUT + "agreement_pct.npy", pct)
ok = fast0 & np.isfinite(pct)
print(T.to_string())
print(f"Consistently reverses (>= 95% of {NB} redraws): {rf.area_share(ok, pct >= 95, grid.LAT):.0f}% of measured area")

# ---- Figure: agreement map (share of redraws in which each location reverses)
import matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from common import NAVY, coast, mapaxes
cls = np.full(pct.shape, np.nan)
cls[~grid.never & ~fast0] = 0
cls[fast0 & (pct < 25)] = 1; cls[fast0 & (pct >= 25) & (pct < 95)] = 2; cls[fast0 & (pct >= 95)] = 3
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20})
fig, ax = plt.subplots(figsize=(12.5, 10.5)); fig.subplots_adjust(left=.12, right=.97, top=.92, bottom=.2)
ax.pcolormesh(grid.LON, grid.LAT, cls, cmap=matplotlib.colors.ListedColormap(["#d9d9d9", "#4575b4", "#f4a582", "#8b0000"]),
              vmin=-.5, vmax=3.5, shading="nearest")
coast(ax, "#333", 1.4); mapaxes(ax)
ax.set_title("Reversal agreement", fontsize=30, fontweight="bold", color=NAVY, pad=14)
fig.legend(handles=[Patch(color="#8b0000", label="Reverses in 95%+ of redraws"), Patch(color="#f4a582", label="Uncertain (25–95%)"),
                    Patch(color="#4575b4", label="Rarely reverses (<25%)"), Patch(color="#d9d9d9", label="Too slow to count")],
           loc="lower center", ncol=2, frameon=False, fontsize=20)
fig.savefig(OUT + "agreement_map.png", dpi=150, facecolor="white"); print("saved", OUT + "agreement_map.png")
