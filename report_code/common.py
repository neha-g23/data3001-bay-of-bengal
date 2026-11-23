"""Shared settings for the report figure scripts.

Data: set the BOB_DATA environment variable to the path of bob_subset.nc,
or put bob_subset.nc in this folder. Outputs go to ./report_outputs/.
"""
import os
import numpy as np
import reversal_figures as rf

DATA = os.environ.get("BOB_DATA", "bob_subset.nc")
OUT = os.environ.get("REPORT_OUT", "report_outputs") + "/"
os.makedirs(OUT, exist_ok=True)
NB = int(os.environ.get("NB", 200))      # bootstrap redraws (set NB=20 for a quick test run)
SEED = 3001
NAVY = "#00325a"


def load():
    """Cleaned hourly observations (same loader and cleaning as reversal_figures.py)."""
    return rf.load_observations(DATA)


def coast(ax, colour="#555", lw=1.5):
    for line in [rf.INDIA_COAST, rf.LANKA_COAST]:
        a = np.asarray(line)
        ax.plot(a[:, 0], a[:, 1], color=colour, lw=lw)


def mapaxes(ax):
    ax.set_xlim(78, 99); ax.set_ylim(5, 23); ax.set_aspect("equal")
    ax.set_xticks([80, 85, 90, 95]); ax.set_yticks([5, 10, 15, 20])
    ax.set_xticklabels(["80°E", "85°E", "90°E", "95°E"]); ax.set_yticklabels(["5°N", "10°N", "15°N", "20°N"])


def coast_distance_km(grid, lines=None):
    """Distance (km) from every grid location to the nearest coastline point."""
    lines = lines or [rf.INDIA_COAST, rf.LANKA_COAST]
    c = np.vstack([rf._densify(np.asarray(l)) for l in lines])
    la, lo = grid.LAT.ravel(), grid.LON.ravel()
    d = np.concatenate([rf.haversine_km(la[i:i+2000], lo[i:i+2000], c[:, 1], c[:, 0]).min(1)
                        for i in range(0, la.size, 2000)])
    return d.reshape(grid.LAT.shape)


def redraw(rng, n):
    """One whole-drifter bootstrap redraw: how many times each drifter is picked."""
    return np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
