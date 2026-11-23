"""Load and clean the drogued/undrogued drifter data for the method comparison scripts.

Set BOB_DATA to the path of bob_subset.nc if it is not in the working directory.
"""
import os
import reversal_figures as rf


def load():
    path = os.environ.get("BOB_DATA", "bob_subset.nc")
    return rf.clean_observations(rf.load_observations(path))
