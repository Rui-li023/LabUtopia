#!/usr/bin/env bash
#
# Thin every captured sub-stage strip down to the delivered frame count and lay
# it out as outputs/paper_task_frames/deliver/<task>/01.png, 02.png, ...
#
# The counts follow the appendix request (one frame per sub-stage for the
# level-4 tasks, ~8 for level 5, 4 per half of a level-3 ID/OOD pair, 6-8 for
# level 1/2). A pair's two halves get the SAME count so their frames line up
# stage-for-stage -- that alignment is the whole point of the comparison.
set -uo pipefail
repo_dir="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
src_root="${SRC_ROOT:-${repo_dir}/outputs/paper_task_frames}"
dst_root="${DST_ROOT:-${src_root}/deliver}"
cd "${repo_dir}" || exit 1

entries=(
  "L4_liquid_mixing:10"
  "L4_open_transport_pour:6"
  "L4_device_operation:7"
  "L5_navigate:8"
  "L5_pick:8"
  "L5_shake:8"
  "L5_pick_place:8"
  "L5_pour:8"
  "L3_open_door_id:4"
  "L3_open_door_ood:4"
  "L3_pour_liquid_id:4"
  "L3_pour_liquid_ood:4"
  "L1_close_centrifuge:6"
  "L1_close_door:6"
  "L1_open_drawer:6"
  "L2_cork_flask:6"
  "L2_flask_to_stand:6"
  "L2_open_close:6"
  "L2_move_pipette:6"
  "L2_remove_stopper:6"
)

for entry in "${entries[@]}"; do
    key="${entry%%:*}"
    count="${entry##*:}"
    if [[ ! -d "${src_root}/${key}/frames" ]]; then
        printf 'MISSING\t%s\n' "${key}"
        continue
    fi
    python experiments/paper/paper_select_frames.py "${src_root}/${key}" "${dst_root}/${key}" --count "${count}" \
        | head -1
done
printf '\nDelivery: %s\n' "${dst_root}"
