#!/usr/bin/env bash
# Rebuild every report figure and table made in report_code/ (writes to report_outputs/).
# Scripts are named after the report section they belong to. Order matters only for
# 4_1_q1_region_table (needs 4_1_q1_headline_bootstrap) and the two scripts that reuse the regions.
set -e
cd "$(dirname "$0")"
for s in \
  3_3_eda_numbers_and_figure 3_3_eda_cell_counts_table \
  3_4_drifter_density_map 3_4_kernel_vs_kriging_diagram 3_4_bandwidth_cv_table \
  4_1_q1_headline_bootstrap 4_1_q1_direction_change_classes 4_1_q1_region_table 4_1_q1_region_map \
  4_2_q2_coast_distance 4_2_q2_speed_turn 4_2_q2_speed_turn_figure 4_2_q2_coast_same_speed \
  4_2_q2_monthly_switch 4_2_q2_monthly_turn_table \
  4_3_q3_sensitivity_table; do
  echo "=== $s"; python "$s.py"
done
