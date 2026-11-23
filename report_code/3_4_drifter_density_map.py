"""Section 3.4, Figure M2: how many different drogued drifters passed through each 0.2-degree square.

Squares with fewer than 3 drifters are shown in green: these are the areas where a
single drifter would dominate a grid-box average, which is why the report uses
kernel smoothing with a minimum-drifter rule instead.
Output: figure_m2_drifter_density.png
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import reversal_figures as rf
from common import load, OUT

NAVY = "#00325a"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.labelsize": 24})
df = load(); dd = df[df.drogue]; res = 0.2
la = np.floor(dd.lat / res).astype(int); lo = np.floor(dd.lon / res).astype(int)
cnt = dd.assign(la=la, lo=lo).groupby(["la", "lo"]).drifter_id.nunique()
ny, nx = 90, 105; i0, j0 = 25, 390                    # 5-23 N, 78-99 E in 0.2-degree steps
G = np.full((ny, nx), np.nan)
for (a, b), n in cnt.items():
    i, j = a - i0, b - j0
    if 0 <= i < ny and 0 <= j < nx: G[i, j] = n
xe = np.linspace(78, 99, nx + 1); ye = np.linspace(5, 23, ny + 1)
fig, ax = plt.subplots(figsize=(12, 10.5)); fig.subplots_adjust(left=.12, right=.88, top=.97, bottom=.1)
pc = ax.pcolormesh(xe, ye, np.where(G >= 3, G, np.nan), cmap="Blues", vmin=0, vmax=float(np.nanpercentile(G, 97)))
ax.pcolormesh(xe, ye, np.where(G < 3, 1.0, np.nan), cmap=matplotlib.colors.ListedColormap(["#b7e4c7"]))
for line in [rf.INDIA_COAST, rf.LANKA_COAST]:
    c = np.asarray(line); ax.plot(c[:, 0], c[:, 1], color="#555", lw=1.5)
ax.set_xlim(78, 99); ax.set_ylim(5, 23); ax.set_aspect("equal")
ax.set_xticks([80, 85, 90, 95]); ax.set_yticks([5, 10, 15, 20])
ax.set_xticklabels(["80°E", "85°E", "90°E", "95°E"]); ax.set_yticklabels(["5°N", "10°N", "15°N", "20°N"])
cb = fig.colorbar(pc, ax=ax, fraction=.045, pad=.03, extend="max"); cb.set_label("Drifters per square")
ax.legend(handles=[Patch(color="#b7e4c7", label="Fewer than 3 drifters")], loc="lower right", fontsize=22)
fig.savefig(OUT + "figure_m2_drifter_density.png", dpi=150, facecolor="white"); print("saved figure_m2_drifter_density.png")
print(f"squares with data: {np.isfinite(G).sum()}, with fewer than 3 drifters: {(G < 3).sum()}")
