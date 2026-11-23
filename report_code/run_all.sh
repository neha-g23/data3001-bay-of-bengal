#!/usr/bin/env bash
# Rebuild every report figure and table (writes to report_outputs/).
set -e
for s in 01_headline_bootstrap 02_direction_change_map 03_region_table 04_region_map 05_monthly_switch \
         06_monthly_by_region 07_q2_coast_distance 08_q2_speed_turn 09_eda_figures 10_east_west_map 11_q2_speed_turn_figure 12_sensitivity_intervals 13_bandwidth_cv 14_kriging_diagram 15_monthly_turn_table 16_q2_coast_same_speed 17_table_e1; do
  echo "=== $s"; python "$s.py"
done
