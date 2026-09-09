#!/usr/bin/env bash
#
# Render the per-task sub-stage figure strips used in the paper appendix.
#
# One entry per figure. Each runs a single scripted rollout headless at 1920x1080
# from a FIXED third-person camera and writes one clean PNG per controller
# sub-stage (see experiments/paper/keyframes.py) into
#   outputs/paper_task_frames/<key>/frames/
# plus a frames.tsv naming the sub-stage each frame belongs to. Frames of failed
# attempts are discarded, so a delivered strip is always one successful rollout.
#
# The camera for every entry was chosen with `experiments/main.py --camera-probe` (which
# renders the scene at rest from a list of candidate eye/target/focal specs in a
# single simulator launch) and is recorded verbatim below so a strip can be
# re-rendered identically.
#
# Usage:
#   experiments/paper/render_paper_task_frames.sh              # every entry, skipping done ones
#   experiments/paper/render_paper_task_frames.sh L4_liquid_mixing L3_open_door_id
#   FORCE=1 experiments/paper/render_paper_task_frames.sh <key>    # re-render even if present
#
# ONE SIMULATOR AT A TIME — a single Isaac Sim instance already saturates the
# GPU, so the entries run strictly in sequence.

set -uo pipefail

repo_dir="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-${HOME}/mambaforge/envs/isaacsim5.1/bin/python}"
output_dir="${OUTPUT_DIR:-${repo_dir}/outputs/paper_task_frames}"
render_width="${RENDER_WIDTH:-1920}"
render_height="${RENDER_HEIGHT:-1080}"
timeout_seconds="${TIMEOUT_SECONDS:-7200}"

if [[ ! -x "${python_bin}" ]]; then
    echo "Isaac Sim Python was not found at ${python_bin}" >&2
    exit 2
fi
mkdir -p "${output_dir}/logs"
cd "${repo_dir}" || exit 1

# key | config | camera "EX EY EZ TX TY TZ" | focal | granularity | seed | extra args
#
# granularity `phase` = one frame per task-level phase; `event` = one per atomic
# state-machine transition (more frames than a figure needs, so the strip is
# thinned afterwards with experiments/paper/paper_select_frames.py — that is deliberate,
# a phase-boundary frame alone misses the mid-phase moment that reads best,
# e.g. the peak pour tilt).
entries=(
  "L4_liquid_mixing|level4_liquid_mixing|2.30 -0.85 1.70 0.08 0.02 0.95|3.0|event|4242|"
  "L4_open_transport_pour|level4_open_transport_pour|-0.55 3.10 2.70 0.20 0.20 1.05|2.7|event|12345|--paper-fixed-phase-order"
  # level4_device_operation does NOT collect in the current tree — every attempt
  # fails at picking_beaker (verified with no publication flags at all), so its
  # strip is rendered by replaying the validated 100-episode dataset instead.
  # See experiments/paper/paper_make_replay_config.py and docs/paper_task_frames.md.
  "L4_device_operation|level4_device_operation_paper_replay|-1.65 2.20 2.30 0.08 0.22 1.05|2.5|phase|0|--config-dir outputs/paper_task_frames/_configs --keyframe-every 110"
  # Level 5 is mobile, and also replayed: its collect path regressed the same way
  # (every attempt fails "object not lifted"). Two extra flags matter here.
  # --paper-camera authors a static /World/PaperCamera because every configured
  # level-5 camera is parented to the Ridgebase, so a world pose written onto one
  # is undone the moment the base drives off. --paper-replay-full plays the whole
  # recording: replay normally stops at the first sustained success, which for
  # shake/pour is the lift — i.e. just before the motion the figure is about.
  #
  # All four share ONE framing recipe so the appendix row reads as a set: aim at
  # (task centre x, 0.28, 0.78), eye 2.10 m away along the fixed bearing
  # (0.855, -0.285, 0.434), focal 2.4. That bearing is the one the first pass
  # used, so the sight line is known to be clear; only the distance (2.81 -> 2.10)
  # and focal (2.0 -> 2.4) changed, for ~1.5x more subject. The per-task centre x
  # is the midpoint of episode 0's actual object placement and base travel, read
  # straight out of the dataset (init_state/object_pose_positions plus
  # robot_world_position + agent_pose[:, 0:2]) rather than estimated:
  #   pick        beaker -3.012, base -3.302 -> -3.236   => -3.15
  #   shake       beaker -1.865, base -1.897 -> -1.994   => -1.93
  #   pick_place  beaker -3.125 / plat -2.672,
  #               base -3.210 -> -2.657                  => -2.94
  #   pour        beaker -2.690 / target -2.436,
  #               base -2.622 -> -2.727                  => -2.58
  # --keyframe-every is set per task so each rollout yields ~15 captured frames
  # (recorded lengths: pick 519, shake 1071, pick_place 789, pour 1035 steps).
  #
  # @scene-lighting: render under the USD scene's OWN authored lights, which is
  # what data collection uses (the collect configs carry no `lighting:` block at
  # all). --paper-lighting instead pins every light to a fixed studio rig
  # (1150 intensity / 0.45 exposure / 5400 K); that was added to stop the white
  # Franka clipping, but at this camera it does not clip, and the rig made the
  # level-5 frames far darker and more saturated than the recorded episodes --
  # visibly not the same lab. It also interacted badly with RTX real-time mode:
  # with no global illumination the HDR dome fill could not shade smoothly and
  # left mottled grey patches on the arm. Dropping the rig fixes both at once.
  # Comparisons: _lighting_compare.png (lights) and _renderer_compare.png (renderer).
  # For reference, PathTracing is cleaner still and shows NO noise even on a
  # moving replay, but costs 6.6 s/step vs ~0.05 s/step -- about 6.5 h for these
  # four tasks -- so RTX real-time stays the default.
  "L5_pick|level5_close_pick_paper_replay|-1.355 -0.319 1.691 -3.150 0.280 0.780|2.4|phase|0|--config-dir outputs/paper_task_frames/_configs --paper-camera --paper-replay-full --keyframe-every 35 @scene-lighting"
  "L5_shake|level5_close_shake_paper_replay|-0.135 -0.319 1.691 -1.930 0.280 0.780|2.4|phase|0|--config-dir outputs/paper_task_frames/_configs --paper-camera --paper-replay-full --keyframe-every 70 @scene-lighting"
  "L5_pick_place|level5_close_pick_place_paper_replay|-1.145 -0.319 1.691 -2.940 0.280 0.780|2.4|phase|0|--config-dir outputs/paper_task_frames/_configs --paper-camera --paper-replay-full --keyframe-every 55 @scene-lighting"
  "L5_pour|level5_close_pour_paper_replay|-0.785 -0.319 1.691 -2.580 0.280 0.780|2.4|phase|0|--config-dir outputs/paper_task_frames/_configs --paper-camera --paper-replay-full --keyframe-every 70 @scene-lighting"
  # Navigation proof shot. The four close_* tasks spawn ~1 m from the dock, so
  # their base barely translates -- across all 100 episodes of each, the largest
  # net base displacement is 0.34 / 0.39 / 0.84 / 0.59 m. No episode of theirs can
  # show "the mobile base drove somewhere". far_pick (retired experiment config,
  # 100 validated episodes, same navigation_lab_01 scene) spawns across the lab:
  # episode 77 is an 11.8 m straight corridor run, x +8.31 -> -3.50, then a turn
  # and the grasp. Camera sits at the far end of the corridor looking back up it,
  # so the robot grows from a speck to full size as it approaches -- that size
  # change is what reads as "it travelled". It drives past the camera at the end,
  # so the last frames are dropped at delivery (--drop 12 13 14 15).
  # Probing history: 6 exterior candidates (the lab is enclosed -- a side camera
  # only sees the outside wall), then 6 over-the-wall, then in-corridor; y=-1.2 at
  # x=9.5 is inside a partition and renders blank white.
  "L5_navigate|level5_far_pick_paper_replay|-8.0 -0.6 3.0 3.0 0.0 0.40|1.5|phase|0|--config-dir outputs/paper_task_frames/_configs --paper-camera --paper-replay-full --keyframe-every 150 @scene-lighting"
  # Level 3 generalization pairs: identical seed, camera and sub-stages; the OOD
  # half swaps each material_paths entry's `materials` for its `test_materials`.
  "L3_open_door_id|level3_open|-0.10 -2.90 2.55 0.25 0.05 1.00|2.5|event|20260830|"
  "L3_open_door_ood|level3_open|-0.10 -2.90 2.55 0.25 0.05 1.00|2.5|event|20260830|--paper-ood-materials"
  # level3_pour_liquid also stopped collecting; replayed from pour_pos/. Both
  # halves replay the SAME episode, so the pair differs only in bench material.
  "L3_pour_liquid_id|level3_pour_liquid_paper_replay|0.00 -2.60 2.40 0.10 -0.10 1.00|2.5|phase|0|--config-dir outputs/paper_task_frames/_configs --paper-replay-full --keyframe-every 90"
  "L3_pour_liquid_ood|level3_pour_liquid_paper_replay|0.00 -2.60 2.40 0.10 -0.10 1.00|2.5|phase|0|--config-dir outputs/paper_task_frames/_configs --paper-replay-full --keyframe-every 90 --paper-ood-materials"
  # Level 1 / Level 2. All replayed from outputs/collect/level{1,2}/<task>/ —
  # several of these configs are also left in `mode: infer` in config/, which
  # blocks on a remote policy server, so the derived replay config is the only
  # way to drive them headless anyway.
  #   lab_001 (close_door / drawers): the drawer sits at z=1.15, hence the
  #   higher aim than the door preset.
  #   lab_002 (centrifuge + glassware): two framings — a wide one for the bulky
  #   centrifuge, a tight one for the small glassware, which is unreadable wide.
  "L1_close_centrifuge|level1_close_centrifuge_paper_replay|0.05 -2.25 2.10 0.00 -0.02 1.00|2.8|phase|0|--config-dir outputs/paper_task_frames/_configs --paper-replay-full --keyframe-every 51"
  "L1_close_door|level1_close_door_paper_replay|-0.10 -2.90 2.55 0.25 0.05 1.00|2.5|phase|0|--config-dir outputs/paper_task_frames/_configs --paper-replay-full --keyframe-every 48"
  "L1_open_drawer|level1_open_drawer_paper_replay|-0.10 -2.90 2.65 0.28 0.05 1.10|2.5|phase|0|--config-dir outputs/paper_task_frames/_configs --paper-replay-full --keyframe-every 80"
  "L2_open_close|level2_open_close_paper_replay|-0.10 -2.90 2.65 0.28 0.05 1.10|2.5|phase|0|--config-dir outputs/paper_task_frames/_configs --paper-replay-full --keyframe-every 120"
  "L2_cork_flask|level2_flask_to_cork_paper_replay|0.17 -1.85 1.85 0.12 -0.02 0.95|3.0|phase|0|--config-dir outputs/paper_task_frames/_configs --paper-replay-full --keyframe-every 70"
  "L2_flask_to_stand|level2_flask_to_triangle_paper_replay|0.17 -1.85 1.85 0.12 -0.02 0.95|3.0|phase|0|--config-dir outputs/paper_task_frames/_configs --paper-replay-full --keyframe-every 71"
  "L2_move_pipette|level2_pipette_to_rack_paper_replay|0.05 -2.25 2.10 0.00 -0.02 1.00|2.8|phase|0|--config-dir outputs/paper_task_frames/_configs --paper-replay-full --keyframe-every 75"
  "L2_remove_stopper|level2_stopper_to_flask_paper_replay|0.17 -1.85 1.85 0.12 -0.02 0.95|3.0|phase|0|--config-dir outputs/paper_task_frames/_configs --paper-replay-full --keyframe-every 76"
)

for entry in "${entries[@]}"; do
    IFS='|' read -r key config camera focal granularity seed extra <<<"${entry}"
    if [[ $# -gt 0 ]] && [[ ! " $* " == *" ${key} "* ]]; then
        continue
    fi
    task_dir="${output_dir}/${key}"
    if [[ -d "${task_dir}/frames" && -z "${FORCE:-}" ]]; then
        printf 'SKIP\t%s\talready rendered\n' "${key}"
        continue
    fi
    rm -rf "${task_dir}"
    seed_args=()
    [[ -n "${seed}" && "${seed}" != "0" ]] && seed_args=(--seed "${seed}")
    # @scene-lighting in an entry's extra args means "render under the USD scene's
    # own authored lights", i.e. exactly what data collection sees. Without it the
    # run gets --paper-lighting, which rewrites every light to a fixed studio rig
    # (1150 intensity / 0.45 exposure / 5400 K) and therefore does NOT match the
    # recorded episodes' appearance.
    lighting_args=(--paper-lighting)
    if [[ "${extra}" == *"@scene-lighting"* ]]; then
        lighting_args=()
        extra="${extra//@scene-lighting/}"
    fi
    printf 'START\t%s\t%s\n' "${key}" "${config}"
    # shellcheck disable=SC2086  # camera and extra are intentionally word-split
    timeout "${timeout_seconds}" "${python_bin}" experiments/main.py \
        --config-name="${config}" \
        --max-episodes 1 --max-attempts 8 \
        --headless --no-video \
        --renderer RayTracedLighting \
        --render-width "${render_width}" --render-height "${render_height}" \
        --single-camera 0 \
        --camera-look-at ${camera} \
        --camera-focal-length "${focal}" \
        ${lighting_args[@]+"${lighting_args[@]}"} \
        "${seed_args[@]}" \
        --keyframe-dir "${task_dir}" \
        --keyframe-granularity "${granularity}" \
        --keyframe-min-steps 25 \
        ${extra} \
        >"${output_dir}/logs/${key}.log" 2>&1
    if [[ -d "${task_dir}/frames" ]]; then
        count="$(find "${task_dir}/frames" -name '*.png' | wc -l)"
        printf 'DONE\t%s\t%s frame(s)\n' "${key}" "${count}"
    else
        printf 'FAIL\t%s\tno successful rollout; see %s/logs/%s.log\n' "${key}" "${output_dir}" "${key}"
    fi
done
