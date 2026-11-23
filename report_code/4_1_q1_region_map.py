"""Direction change map with regions A-F outlined and labelled with their reversal share.

Needs regions.npy and region_table.csv from 4_1_q1_region_table.py.
Output: region_map.png
"""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import reversal_figures as rf
from common import load, OUT, NAVY, coast, mapaxes
from importlib import import_module

turn_classes = import_module("4_1_q1_direction_change_classes").turn_classes
COLS = import_module("4_1_q1_direction_change_classes").COLS
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.labelsize": 24})

df = load(); grid = rf.Grid(0.2, df)
cls, _ = turn_classes(df, grid)
M = np.load(OUT + "regions.npy"); T = pd.read_csv(OUT + "region_table.csv")
tags = ["A", "B", "C", "D", "E", "F"]
pos = [(84.6, 18.6), (79.7, 6.3), (88.0, 6.9), (84.8, 13.4), (92.0, 12.4), (90.0, 19.6)]   # label positions
fig, ax = plt.subplots(figsize=(13.5, 11.2)); fig.subplots_adjust(left=.11, right=.97, top=.92, bottom=.24)
ax.pcolormesh(grid.LON, grid.LAT, cls, cmap=matplotlib.colors.ListedColormap(COLS), vmin=-.5, vmax=3.5,
              shading="nearest", alpha=.8)
for k in range(6):
    ax.contour(grid.LON, grid.LAT, (M[k] & ~grid.never).astype(float), levels=[.5], colors=NAVY, linewidths=2.5)
    ax.text(*pos[k], f"{tags[k]}\n{T.reverses_pct[k]}%", ha="center", va="center", fontsize=21, fontweight="bold",
            color=NAVY, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=NAVY, alpha=.92))
coast(ax); mapaxes(ax)
ax.set_title("Reversal by region", fontsize=30, fontweight="bold", color=NAVY, pad=14)
fig.legend(handles=[Patch(color=COLS[3], alpha=.8, label="Near-full reversal (>135°)"),
                    Patch(color=COLS[2], alpha=.8, label="Reverses (90–135°)"),
                    Patch(color=COLS[1], alpha=.8, label="No reversal (under 90°)"),
                    Patch(color=COLS[0], alpha=.8, label="Too slow to count")],
           loc="lower center", bbox_to_anchor=(.5, .05), ncol=2, frameon=False, fontsize=19)
fig.text(.5, .02, "Labels: % of each region's measured area that turns more than 90°", ha="center", fontsize=17, color="#444")
fig.savefig(OUT + "region_map.png", dpi=150, facecolor="white"); print("saved", OUT + "region_map.png")
