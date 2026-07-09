#!/usr/bin/env bash
# Export the four Level-5 mobile-manipulation tasks to LeRobot v2.1, each as a
# SEPARATE dataset (no merge). Source = the fixed-nav + new-camera full-100
# collect from 2026-07-07. Robot = ridgebase (11-dim base+arm+gripper state).
set -e
cd /home/ubuntu/Documents/LabUtopia
PY=~/mambaforge/envs/isaacsim5.1/bin/python
DST_ROOT=outputs/lerobot/v21_level5

TASKS=(close_pick far_pick close_pick_place far_transport_place)
# all four re-collected 2026-07-09 with the carry-crab fix + revolute-joint
# facing spawn + ~3cm dock-position jitter (VLA-robustness generalization).
SRC_close_pick=outputs/collect/2026.07.09/00.29.44_level5_close_pick
SRC_far_pick=outputs/collect/2026.07.09/00.52.42_level5_far_pick
SRC_close_pick_place=outputs/collect/2026.07.09/01.50.18_level5_close_pick_place
SRC_far_transport_place=outputs/collect/2026.07.09/02.23.34_level5_far_transport_place

for task in "${TASKS[@]}"; do
  eval "src=\$SRC_$task"
  dst="$DST_ROOT/$task"
  echo "############### ▶ $task : $src → $dst  $(date '+%T') ###############"
  $PY -m scripts.lerobot_export.cli --src "$src" --dst "$dst" --version v2.1 --robot ridgebase --base-action body_delta
done

echo "############### L5 LeRobot EXPORT DONE  $(date '+%T') ###############"
du -sh "$DST_ROOT"/* 2>/dev/null
