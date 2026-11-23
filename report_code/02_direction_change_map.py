"""Direction change map (Q1): how far the current turns between the Southwest
(Jun-Sep) and Northeast (Dec-Feb) monsoons, in four classes.

Output: direction_change_map.png. Prints the area shares
(<=90, 90-135, >135 degrees) of the measured Bay.
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import reversal_figures as rf
from common import load, OUT, NAVY, coast, mapaxes

COLS = ["#d9d9d9", "#fdd9b5", "#f16913", "#7f0000"]   # too slow, <90, 90-135, >135
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 20, "axes.labelsize": 24})


def turn_classes(df, grid):
    """0 = too slow, 1 = no reversal (<=90), 2 = reverses (90-135), 3 = near-full (>135)."""
    fields, _ = rf.seasonal_fields(df, "drogued", grid)
    A = rf.seasonal_angle(fields, grid)
    ang, fast, both = A["angle"], A["fast"], A["both"]
    cls = np.full(ang.shape, np.nan)
    cls[both & ~fast] = 0
    cls[fast & (ang <= 90)] = 1; cls[fast & (ang > 90) & (ang <= 135)] = 2; cls[fast & (ang > 135)] = 3
    return cls, A


if __name__ == "__main__":
    df = load(); grid = rf.Grid(0.2, df)
    cls, A = turn_classes(df, grid)
    ang, fast = A["angle"], A["fast"]
    sh = [rf.area_share(fast, c, grid.LAT) for c in (ang <= 90, (ang > 90) & (ang <= 135), ang > 135)]
    print("Area shares: <=90: %.1f%%, 90-135: %.1f%%, >135: %.1f%%, reverses (>90): %.1f%%" % (*sh, sh[1] + sh[2]))
    fig, ax = plt.subplots(figsize=(12.5, 10.8)); fig.subplots_adjust(left=.12, right=.97, top=.92, bottom=.22)
    ax.pcolormesh(grid.LON, grid.LAT, cls, cmap=matplotlib.colors.ListedColormap(COLS), vmin=-.5, vmax=3.5, shading="nearest")
    coast(ax); mapaxes(ax)
    ax.set_title("Direction change", fontsize=30, fontweight="bold", color=NAVY, pad=14)
    fig.legend(handles=[Patch(color=COLS[3], label="Near-full reversal (>135°)"),
                        Patch(color=COLS[2], label="Reverses (90–135°)"),
                        Patch(color=COLS[1], label="No reversal (under 90°)"),
                        Patch(color=COLS[0], label="Too slow to count (< 0.1 m/s)")],
               loc="lower center", ncol=2, frameon=False, fontsize=19)
    fig.savefig(OUT + "direction_change_map.png", dpi=150, facecolor="white")
    print("saved", OUT + "direction_change_map.png")
