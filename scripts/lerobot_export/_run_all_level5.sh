#!/usr/bin/env bash
# Export the four Level-5 mobile-manipulation tasks to LeRobot v2.1, each as a
# SEPARATE dataset (no merge). Source = the fixed-nav + new-camera full-100
# collect from 2026-07-07. Robot = ridgebase (11-dim base+arm+gripper state).
set -e
cd /home/ubuntu/Documents/LabUtopia
PY=~/mambaforge/envs/isaacsim5.1/bin/python
DST_ROOT=outputs/lerobot/v21_level5

TASKS=(close_pick far_pick close_pick_place far_transport_place)
# close_* = re-collected 2026-07-08 with the ~1 m facing-object spawn + the
# revolute-joint facing fix (no crab); far_* = the 2026-07-07 full-100 collect.
SRC_close_pick=outputs/collect/2026.07.08/14.02.50_level5_close_pick
SRC_far_pick=outputs/collect/2026.07.07/18.59.12_level5_far_pick
SRC_close_pick_place=outputs/collect/2026.07.08/14.23.38_level5_close_pick_place
SRC_far_transport_place=outputs/collect/2026.07.07/20.46.28_level5_far_transport_place

for task in "${TASKS[@]}"; do
  eval "src=\$SRC_$task"
  dst="$DST_ROOT/$task"
  echo "############### ▶ $task : $src → $dst  $(date '+%T') ###############"
  $PY -m scripts.lerobot_export.cli --src "$src" --dst "$dst" --version v2.1 --robot ridgebase
done

echo "############### L5 LeRobot EXPORT DONE  $(date '+%T') ###############"
du -sh "$DST_ROOT"/* 2>/dev/null
