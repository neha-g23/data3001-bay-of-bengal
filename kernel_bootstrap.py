"""
Drifter bootstrap for the Gaussian-kernel-smoothed circulation maps (DATA3001).

Why this exists
---------------
project3001.py bootstraps the *raw* 1-degree cell means (standard errors, and
the confidence interval on the percentage of cells that reverse). The kernel
smoothing section only bootstraps the held-out RMSE improvement. Nothing
describes how uncertain the smoothed maps themselves are, and project3001.py
says so in kernel_summary.txt ("Original bootstrap uncertainty does not
describe the smoothed estimator"). This script fills that gap.

What it does
------------
Each bootstrap draw resamples WHOLE DRIFTERS with replacement (a drifter drawn
twice counts twice, one drifter = one vote, as in the rest of the project),
rebuilds the cell means, applies the same Gaussian kernel smoother, and
records the result. From N_BOOT draws it reports:

1. Per cell: bootstrap standard error and 95% interval of the smoothed
   velocity and speed, a direction-stability measure, and the same quantities
   for the unsmoothed cell mean, estimated from the SAME draws so the two are
   directly comparable.
2. Monsoon reversal (SW vs NE direction differs by more than 90 / 135 degrees):
   the percentage of cells reversed, with a bootstrap 95% interval, for the raw
   cell means and for the smoothed field at several bandwidths, plus the
   paired difference (smoothed minus raw). Reported for the whole Bay and the
   EICC band, with and without a weak-current filter.

Within one draw the reversal percentage is computed over the cells that are
mapped (>= MIN_DRIFTERS drifters) in BOTH seasons in that draw, exactly as
project3001.bootstrap_reversal does, and the raw and smoothed values use the
same cells so that they are comparable.

Which interval to report for the reversal percentage
----------------------------------------------------
Two intervals are saved. In a simulation with a known field (random-walk
drifters, correlated errors, about half the cells reversed) the bootstrap
standard error of the reversal percentage was about right (1.08x the true
sampling SD), but the bootstrap distribution sat 2 to 3 percentage points BELOW
the estimate. A percentile interval inherits that shift and covered the
estimator's sampling mean only 87% (raw) and 77% (smoothed) of the time at a
nominal 95%. Estimate +/- 1.96 bootstrap SE covered 95% and 93%, so it is the
one labelled *_ci95_* here. The same downward shift is visible in the real
results: 66.9% estimate vs 64.5% bootstrap median for the whole Bay. The
percentile interval is kept (*_percentile_*) for comparison with project3001.py.
Neither interval corrects the estimator's own bias: noisy mean currents push
the measured reversal percentage towards 50%, which a bootstrap cannot detect.

Limitations (also worth stating in the report)
----------------------------------------------
* The bandwidth is held fixed at the cross-validated value in every draw.
  The intervals therefore do not include uncertainty from choosing the
  bandwidth. The bandwidth sensitivity rows show how much that choice matters.
* The bootstrap measures sampling variability (which drifters happened to pass
  through), not bias from smoothing across coastlines or sharp fronts.
* With very few drifters in a cell the bootstrap understates uncertainty, so
  treat intervals for cells with fewer than HIGH_CONF_DRIFTERS drifters as
  optimistic.

How to run
----------
    uv run python project3001.py          # first, to create bob_subset.nc and
                                          # outputs/tables/kernel_bandwidth_cv.csv
    uv run python kernel_bootstrap.py     # then this (about a minute)

Options:  --n-boot 2000   --bandwidth KM   --seed 3001
If outputs/tables/kernel_bandwidth_cv.csv does not exist, pass --bandwidth.
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd

import project3001 as p

DEFAULT_N_BOOT = 2000
DEFAULT_SEED = 3001
CUTOFFS = (0.0, 0.05)                    # m/s; 0.05 matches MIN_SPEED_FOR_PCT / the weak-current check
THRESHOLDS = (90, 135)                   # degrees
SENSITIVITY_BANDWIDTHS_KM = (50.0, 100.0, 175.0, 250.0)


# ------------------------------------------------------------
# Building blocks (kept small and importable so they can be tested)
# ------------------------------------------------------------

def prepare(per):
    """Index the per-drifter cell means so that cell means can be rebuilt
    quickly for any set of drifter weights.

    Only drifter-cell records with >= MIN_OBS_PER_DRIFTER hours are used,
    exactly as in project3001.cell_grid. Seasons are indexed 0 = Southwest,
    1 = Northeast (the order of project3001.SEASONS)."""
    q = per[per["n_obs"] >= p.MIN_OBS_PER_DRIFTER]
    q = q[q["monsoon"].isin(p.SEASONS)]
    ids, drifter_idx = np.unique(q["drifter_id"].to_numpy(), return_inverse=True)
    cells, cell_idx = np.unique(q[["lon0", "lat0"]].to_numpy(), axis=0, return_inverse=True)
    season_idx = q["monsoon"].map({s: i for i, s in enumerate(p.SEASONS)}).to_numpy()
    lon0, lat0 = cells[:, 0], cells[:, 1]
    return {
        "n_drifters": len(ids),
        "drifter_idx": drifter_idx.ravel(),
        "slot": cell_idx.ravel() * 2 + season_idx,        # one slot per (cell, season)
        "n_cells": len(cells),
        "ve": q["ve"].to_numpy(dtype=float),
        "vn": q["vn"].to_numpy(dtype=float),
        "lon0": lon0,
        "lat0": lat0,
        "lon": lon0 + p.GRID / 2,                          # cell centres, as in project3001
        "lat": lat0 + p.GRID / 2,
        "regions": {
            "Whole Bay of Bengal": np.ones(len(cells), dtype=bool),
            "Approx. EICC band 80-85E": np.asarray(p.in_eicc(lon0), dtype=bool),
        },
    }


def cell_means(prep, drifter_weights):
    """Drifter-weighted cell means for one resample.

    drifter_weights[i] = how many times drifter i was drawn (all ones gives
    the original data). Returns (ve, vn, sufficient), each shaped
    (n_cells, 2). A cell is sufficient in a season if at least MIN_DRIFTERS
    DISTINCT drifters were drawn there."""
    w = drifter_weights[prep["drifter_idx"]]
    slot, n_slots = prep["slot"], prep["n_cells"] * 2
    w_sum = np.bincount(slot, w, n_slots)
    n_distinct = np.bincount(slot, (w > 0).astype(float), n_slots)
    suff = (n_distinct >= p.MIN_DRIFTERS).reshape(-1, 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        ve = (np.bincount(slot, w * prep["ve"], n_slots) / w_sum).reshape(-1, 2)
        vn = (np.bincount(slot, w * prep["vn"], n_slots) / w_sum).reshape(-1, 2)
    return np.where(suff, ve, 0.0), np.where(suff, vn, 0.0), suff


def kernel_matrix(lon, lat, h_km):
    """Gaussian weights between every pair of cell centres. h_km = 0 gives the
    identity, i.e. no smoothing (each cell keeps its own mean). Identical to
    the weights in project3001.kernel_predict, including the 3h cut-off."""
    n = len(lon)
    if h_km <= 0:
        return np.eye(n)
    d = p.kernel_distance_km(lon, lat, lon, lat)
    w = np.exp(-0.5 * (d / h_km) ** 2)
    w[d > p.KERNEL_RADIUS_MULTIPLIER * h_km] = 0.0
    return w


def smooth(weights, ve, vn, suff):
    """Kernel-smoothed velocity at every cell, using only the cells that are
    sufficient in that season as reference points (as in
    project3001.kernel_predict). NaN where no reference cell is in range."""
    s = suff.astype(float)
    den = weights @ s
    with np.errstate(invalid="ignore", divide="ignore"):
        ue = np.where(den > 0, (weights @ (s * ve)) / den, np.nan)
        un = np.where(den > 0, (weights @ (s * vn)) / den, np.nan)
    return ue, un


def reversal_pcts(ue, un, cell_mask, min_speed):
    """Percentage of cells whose SW-to-NE direction change exceeds each
    threshold, plus the number of cells used. Cells where either season's
    speed is below min_speed are ignored (their direction is noise)."""
    angle = p.angle_between(ue[:, 0], un[:, 0], ue[:, 1], un[:, 1])
    slowest = np.minimum(np.hypot(ue[:, 0], un[:, 0]), np.hypot(ue[:, 1], un[:, 1]))
    ok = cell_mask & np.isfinite(angle) & (slowest >= min_speed)
    if not ok.any():
        return [np.nan] * len(THRESHOLDS) + [0]
    return [float((angle[ok] > t).mean() * 100) for t in THRESHOLDS] + [int(ok.sum())]


def run_draws(prep, bandwidths, n_boot, seed, store_for=()):
    """Run the whole-drifter bootstrap.

    Returns
      stored: {h: array (n_boot, n_cells, 2 seasons, 2 components)} of
              smoothed velocities, for the bandwidths in store_for
      rev:    {(h, region, min_speed): array (n_boot, 3)} of
              [pct > 90, pct > 135, n cells] for every draw
    """
    n, n_dr = prep["n_cells"], prep["n_drifters"]
    mats = {h: kernel_matrix(prep["lon"], prep["lat"], h) for h in bandwidths}
    stored = {h: np.full((n_boot, n, 2, 2), np.nan) for h in store_for}
    keys = list(itertools.product(bandwidths, prep["regions"], CUTOFFS))
    rev = {k: np.full((n_boot, len(THRESHOLDS) + 1), np.nan) for k in keys}

    rng = np.random.default_rng(seed)
    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n_dr, n_dr), minlength=n_dr)
        ve, vn, suff = cell_means(prep, w)
        both = suff.all(axis=1)                     # mapped in both seasons in THIS draw
        for h, mat in mats.items():
            ue, un = smooth(mat, ve, vn, suff)
            if h in stored:
                stored[h][b, :, :, 0], stored[h][b, :, :, 1] = ue, un
            for region, rmask in prep["regions"].items():
                for cutoff in CUTOFFS:
                    rev[(h, region, cutoff)][b] = reversal_pcts(ue, un, both & rmask, cutoff)
    return stored, rev


def point_estimates(prep, bandwidths):
    """Estimates from the original (un-resampled) data."""
    ve, vn, suff = cell_means(prep, np.ones(prep["n_drifters"]))
    both = suff.all(axis=1)
    est = {h: smooth(kernel_matrix(prep["lon"], prep["lat"], h), ve, vn, suff) for h in bandwidths}
    rev = {(h, region, cutoff): reversal_pcts(*est[h], both & rmask, cutoff)
           for h in bandwidths for region, rmask in prep["regions"].items() for cutoff in CUTOFFS}
    return ve, vn, suff, est, rev


def percentile_ci(x, q=(2.5, 50, 97.5)):
    x = x[np.isfinite(x)]
    return tuple(np.percentile(x, q)) if len(x) else (np.nan,) * len(q)


# ------------------------------------------------------------
# Tables
# ------------------------------------------------------------

def cell_table(prep, ve, vn, suff, est, stored, h_best):
    """One row per mapped (cell, season): point estimates and bootstrap
    uncertainty for the smoothed field and, from the same draws, for the raw
    cell mean."""
    sm, raw = stored[h_best], stored[0.0]
    rows = []
    for s, season in enumerate(p.SEASONS):
        for i in np.flatnonzero(suff[:, s]):
            ue, un = est[h_best][0][i, s], est[h_best][1][i, s]
            sm_ve, sm_vn = sm[:, i, s, 0], sm[:, i, s, 1]
            ok = np.isfinite(sm_ve)
            sm_speed = np.hypot(sm_ve, sm_vn)
            raw_ve, raw_vn = raw[:, i, s, 0], raw[:, i, s, 1]
            raw_ok = np.isfinite(raw_ve)
            raw_speed = np.hypot(raw_ve, raw_vn)
            dev = p.angle_between(sm_ve[ok], sm_vn[ok], ue, un)
            lo, hi = percentile_ci(sm_speed)[0], percentile_ci(sm_speed)[2]
            se_sm = np.nanstd(sm_speed, ddof=1)
            se_raw = np.nanstd(raw_speed, ddof=1) if raw_ok.sum() > 1 else np.nan
            rows.append({
                "monsoon": season, "lon0": prep["lon0"][i], "lat0": prep["lat0"][i],
                "ve_raw": ve[i, s], "vn_raw": vn[i, s],
                "ve_smooth": ue, "vn_smooth": un, "speed_smooth": float(np.hypot(ue, un)),
                "se_ve_smooth": np.nanstd(sm_ve, ddof=1), "se_vn_smooth": np.nanstd(sm_vn, ddof=1),
                "se_speed_smooth": se_sm,
                "speed_ci95_low": lo, "speed_ci95_high": hi,
                "dir_dev95_deg": float(np.percentile(dev, 95)) if len(dev) else np.nan,
                "se_speed_raw_sameDraws": se_raw,
                "raw_defined_pct_of_draws": float(raw_ok.mean() * 100),
                "se_ratio_smooth_over_raw": se_sm / se_raw if se_raw and se_raw > 0 else np.nan,
            })
    return pd.DataFrame(rows)


def normal_ci(estimate, draws):
    """estimate +/- 1.96 bootstrap standard errors, kept within 0-100 (percent)."""
    se = float(np.nanstd(draws, ddof=1))
    return se, max(0.0, estimate - 1.96 * se), min(100.0, estimate + 1.96 * se)


def reversal_table(point, rev, bandwidths):
    """Reversal percentages with two kinds of 95% interval:
      *_ci95_*        estimate +/- 1.96 bootstrap SE (recommended; see module docstring)
      *_percentile_*  2.5th / 97.5th percentile of the draws (what project3001.py reports)
    and, for smoothed rows, the paired difference smoothed minus raw."""
    rows = []
    for (h, region, cutoff), draws in rev.items():
        pt = point[(h, region, cutoff)]
        row = {"bandwidth_km": h, "region": region, "min_speed_mps": cutoff, "cells": pt[2]}
        for j, t in enumerate(THRESHOLDS):
            lo, med, hi = percentile_ci(draws[:, j])
            se, nlo, nhi = normal_ci(pt[j], draws[:, j])
            row.update({f"pct_gt{t}": pt[j], f"pct_gt{t}_boot_se": se,
                        f"pct_gt{t}_ci95_low": nlo, f"pct_gt{t}_ci95_high": nhi,
                        f"pct_gt{t}_boot_median": med,
                        f"pct_gt{t}_percentile_low": lo, f"pct_gt{t}_percentile_high": hi})
            if h > 0:                               # paired difference against the raw cell means
                raw_draws = rev[(0.0, region, cutoff)][:, j]
                diff = pt[j] - point[(0.0, region, cutoff)][j]
                dse = float(np.nanstd(draws[:, j] - raw_draws, ddof=1))
                row.update({f"diff_vs_raw_gt{t}": diff, f"diff_vs_raw_gt{t}_boot_se": dse,
                            f"diff_vs_raw_gt{t}_ci95_low": diff - 1.96 * dse,
                            f"diff_vs_raw_gt{t}_ci95_high": diff + 1.96 * dse})
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.sort_values(["region", "min_speed_mps", "bandwidth_km"], ignore_index=True)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def best_bandwidth(arg):
    if arg is not None:
        return float(arg)
    path = os.path.join(p.TAB_DIR, "kernel_bandwidth_cv.csv")
    if not os.path.exists(path):
        raise SystemExit("Cannot find outputs/tables/kernel_bandwidth_cv.csv. Run project3001.py "
                         "first, or pass the bandwidth with --bandwidth KM.")
    cv = pd.read_csv(path)
    cv = cv[(cv["scope"] == "Overall") & (cv["bandwidth_km"] > 0)].sort_values(["rmse_vector_mps", "bandwidth_km"])
    return float(cv.iloc[0]["bandwidth_km"])


def load_per_drifter():
    """The same drogued, SW/NE-only per-drifter table that project3001.run() builds."""
    ds = p.load_subset()
    obs, _ = p.clean_obs(p.build_obs_table(ds))
    season_obs = obs[obs["monsoon"].isin(p.SEASONS)]
    return p.per_drifter_means(season_obs[season_obs["drogue"]])


def check_against_project(point):
    """Sanity check: the raw point estimates here should reproduce
    outputs/tables/reversal_summary.csv (drogued only)."""
    path = os.path.join(p.TAB_DIR, "reversal_summary.csv")
    if not os.path.exists(path):
        return
    ref = pd.read_csv(path)
    ref = ref[ref["dataset"] == "drogued only"].set_index("region")
    for region in ref.index:
        mine = point[(0.0, region, 0.0)]
        ok = abs(mine[0] - ref.loc[region, "reversed_pct"]) < 1e-6 and mine[2] == ref.loc[region, "cells_both_seasons"]
        print(f"  check vs reversal_summary.csv, {region}: "
              f"{'OK' if ok else 'MISMATCH'} ({mine[0]:.1f}% of {mine[2]} cells here, "
              f"{ref.loc[region, 'reversed_pct']:.1f}% of {ref.loc[region, 'cells_both_seasons']} there)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    ap.add_argument("--bandwidth", type=float, default=None, help="smoothing bandwidth in km (default: CV choice)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = ap.parse_args()

    os.makedirs(p.FIG_DIR, exist_ok=True)
    os.makedirs(p.TAB_DIR, exist_ok=True)

    p.section("BOOTSTRAP OF THE KERNEL-SMOOTHED FIELD (drogued only)")
    h_best = best_bandwidth(args.bandwidth)
    bandwidths = sorted({0.0, h_best, *SENSITIVITY_BANDWIDTHS_KM})
    prep = prepare(load_per_drifter())
    print(f"Drifters: {prep['n_drifters']}, cells: {prep['n_cells']}, bandwidth h = {h_best:g} km, "
          f"{args.n_boot} whole-drifter draws, seed {args.seed}")

    ve, vn, suff, est, point = point_estimates(prep, bandwidths)
    check_against_project(point)

    stored, rev = run_draws(prep, bandwidths, args.n_boot, args.seed, store_for=(0.0, h_best))

    cells = cell_table(prep, ve, vn, suff, est, stored, h_best)
    p.save_table(cells, "kernel_bootstrap_cells.csv")
    reversal = reversal_table(point, rev, bandwidths)
    p.save_table(reversal, "kernel_bootstrap_reversal.csv")

    print("\nSMOOTHED-FIELD UNCERTAINTY (per mapped cell and season)")
    for season, g in cells.groupby("monsoon"):
        print(f"  {season}: median SE of speed, raw {g['se_speed_raw_sameDraws'].median():.4f} m/s, "
              f"smoothed {g['se_speed_smooth'].median():.4f} m/s; "
              f"median 95th-percentile direction deviation {g['dir_dev95_deg'].median():.0f} deg; "
              f"cells with direction deviation > 45 deg: {(g['dir_dev95_deg'] > 45).sum()} of {len(g)}")

    print("\nMONSOON REVERSAL, % of cells with SW-to-NE direction change above threshold")
    show = reversal[reversal["bandwidth_km"].isin([0.0, h_best])]
    cols = ["region", "min_speed_mps", "bandwidth_km", "cells", "pct_gt90", "pct_gt90_ci95_low",
            "pct_gt90_ci95_high", "pct_gt135", "pct_gt135_ci95_low", "pct_gt135_ci95_high"]
    show = show[cols].round(1).assign(min_speed_mps=show["min_speed_mps"].map("{:.2f}".format))
    print("(bandwidth 0 = raw cell means; intervals are estimate +/- 1.96 bootstrap SE)")
    print(show.to_string(index=False))
    print("\nBANDWIDTH SENSITIVITY (whole Bay, no speed filter), > 90 deg")
    sens = reversal[(reversal["region"] == "Whole Bay of Bengal") & (reversal["min_speed_mps"] == 0.0)]
    print(sens[["bandwidth_km", "pct_gt90", "pct_gt90_ci95_low", "pct_gt90_ci95_high"]].round(1).to_string(index=False))

    figure(reversal, cells, h_best)


def figure(reversal, cells, h_best):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    ax = axes[0]
    colours = {r: c for r, c in zip(sorted(reversal["region"].unique()), ("#2166ac", "#b2182b"))}
    for (region, cutoff), g in reversal.groupby(["region", "min_speed_mps"]):
        g = g.sort_values("bandwidth_km")
        x = np.arange(len(g))
        ax.plot(x, g["pct_gt90"], marker="o", color=colours[region], ls="-" if cutoff == 0 else "--",
                label=f"{region}, speed >= {cutoff:g} m/s")
        if cutoff == 0:                              # shade the interval for the unfiltered series only
            ax.fill_between(x, g["pct_gt90_ci95_low"], g["pct_gt90_ci95_high"], color=colours[region], alpha=0.15)
    ax.set_xticks(np.arange(len(g)))
    ax.set_xticklabels([("raw cells" if b == 0 else f"{b:g} km") for b in g["bandwidth_km"]])
    ax.axhline(50, color="grey", lw=0.8, ls=":")
    ax.set_ylabel("% of cells with SW-to-NE change > 90 deg")
    ax.set_title("Reversal percentage vs smoothing bandwidth (95% bootstrap interval)")
    ax.legend(fontsize=8)

    ax = axes[1]
    for season, g in cells.groupby("monsoon"):
        ax.scatter(g["se_speed_raw_sameDraws"], g["se_speed_smooth"], s=14, alpha=0.6, label=season)
    top = np.nanmax(cells[["se_speed_raw_sameDraws", "se_speed_smooth"]].to_numpy())
    ax.plot([0, top], [0, top], color="grey", lw=0.8, ls="--")
    ax.set_xlabel("Bootstrap SE of speed, raw cell mean (m/s)")
    ax.set_ylabel(f"Bootstrap SE of speed, smoothed h = {h_best:g} km (m/s)")
    ax.set_title("Smoothing reduces sampling noise (below the line)")
    ax.legend()
    p.save_fig(fig, "14_kernel_bootstrap.png", dpi=200)


if __name__ == "__main__":
    main()