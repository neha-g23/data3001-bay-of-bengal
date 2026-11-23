"""East-west current map, drogued drifters (Southwest vs Northeast monsoon).

Colour = east-west speed (orange east, blue west), arrows = direction and speed,
grey = fewer than 3 drifters within 100 km. Prints the share of the mapped area
flowing east in each monsoon.
Output: east_west_drogued.png
"""
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, TwoSlopeNorm
import reversal_figures as rf
from common import load, OUT

df = load(); grid = rf.Grid(0.2, df)
fields, _ = rf.seasonal_fields(df, "drogued", grid)
LON, LAT = grid.LON, grid.LAT
cmap = LinearSegmentedColormap.from_list("west_east", ["#08306b", "#2171b5", "#9ecae1", "#f7f7f7", "#fdb863", "#e66101", "#7f2704"])
norm = TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)
fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.8))
fig.subplots_adjust(left=0.05, right=0.86, top=0.86, bottom=0.13, wspace=0.08)
st = (slice(2, None, 5), slice(2, None, 5))     # one arrow per degree
ext = [78, 99, 5, 23]
for ax, s, lab in zip(axes, ["Southwest", "Northeast"], ["a", "b"]):
    ve, vn, low = fields[s]
    ax.imshow(np.where(low, 1.0, np.nan), origin="lower", extent=ext, cmap=ListedColormap(["#e3e3e3"]), interpolation="nearest")
    im = ax.imshow(ve, origin="lower", extent=ext, cmap=cmap, norm=norm, interpolation="bilinear")
    q = ax.quiver(LON[st], LAT[st], ve[st], vn[st], color="#222222", scale=7, width=0.0032, headwidth=4)
    ok = np.isfinite(ve); w = np.cos(np.radians(LAT))[ok]
    east = (w * (ve[ok] > 0)).sum() / w.sum() * 100
    ax.set_xlim(78, 99); ax.set_ylim(5, 23); ax.set_aspect("equal")
    ax.set_title(f"({lab}) {s} monsoon: {east:.0f}% of mapped area flows east", fontsize=13)
    ax.set_xlabel("Longitude (°E)"); print(s, "flows east:", round(east, 1), "%")
axes[0].set_ylabel("Latitude (°N)")
axes[1].quiverkey(q, 0.80, -0.09, 0.5, "0.5 m/s", labelpos="E", coordinates="axes")
cax = fig.add_axes([0.88, 0.14, 0.018, 0.66])
cb = fig.colorbar(im, cax=cax, extend="both", ticks=[-0.5, -0.25, 0, 0.25, 0.5]); cb.set_label("East-west current (m/s)", fontsize=11)
cax.text(0.5, 1.03, "Eastward", transform=cax.transAxes, ha="center", va="bottom", color="#e66101", fontsize=10, fontweight="bold")
cax.text(0.5, -0.03, "Westward", transform=cax.transAxes, ha="center", va="top", color="#2171b5", fontsize=10, fontweight="bold")
fig.text(0.05, 0.02, "Arrows: current direction and speed. Grey: fewer than 3 drifters within 100 km.", fontsize=9)
fig.suptitle("East-west surface currents reverse between monsoons (kernel-smoothed, drogued drifters, 75 km bandwidth)", fontsize=14)
fig.savefig(OUT + "east_west_drogued.png", dpi=140); print("saved", OUT + "east_west_drogued.png")
