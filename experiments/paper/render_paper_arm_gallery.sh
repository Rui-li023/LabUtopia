#!/usr/bin/env bash
#
# Render one publication frame per supported arm: the arm standing on the
# lab_003 bench, holding the beaker it just picked, shot head-on.
#
# Scene composition
# -----------------
# Physics stays on lab_001 -- that is the scene every one of these 24 configs
# was tuned and validated against (see outputs/all_arm_pick_renders/manifest.tsv,
# 24/24 success). Only the *look* is swapped: lab_001's work surface is hidden
# with --paper-hide-prims (visibility is render-only, so its colliders still
# carry the arm and the cup), and lab_003 is referenced as a backdrop whose
# /World/table top lands at the same height (0.712 m vs 0.713 m, plus the 8 mm
# lift below that puts it flush with lab_001's Cube collider at 0.720 m).
# The result is a real lab bench and a real lab room behind the arm, with zero
# risk to any tuned grasp.
#
# Camera
# ------
# Head-on from +X at y=0, pushed back to 3.4 m and tilted ~11 degrees down so the
# bench's front edge stays in frame and the tallest arms still fit. Fixed across
# every arm so the 24 panels are directly comparable.
#
# Usage: experiments/paper/render_paper_arm_gallery.sh
#   OUTPUT_DIR=... PYTHON_BIN=... RENDER_WIDTH=... RENDER_HEIGHT=... to override.
#   Already-rendered arms are skipped, so the sweep is resumable.

set -uo pipefail

repo_dir="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-${HOME}/mambaforge/envs/isaacsim5.1/bin/python}"
output_dir="${OUTPUT_DIR:-${repo_dir}/outputs/paper_arm_gallery}"
render_width="${RENDER_WIDTH:-3840}"
render_height="${RENDER_HEIGHT:-2160}"
timeout_seconds="${TIMEOUT_SECONDS:-2400}"
max_attempts="${MAX_ATTEMPTS:-3}"

if [[ ! -x "${python_bin}" ]]; then
    echo "Isaac Sim Python was not found at ${python_bin}" >&2
    echo "Set PYTHON_BIN to the Isaac Sim 5.1 Python executable." >&2
    exit 2
fi

mkdir -p "${output_dir}/logs"
cd "${repo_dir}" || exit 1

manifest="${output_dir}/manifest.tsv"
if [[ ! -f "${manifest}" ]]; then
    printf 'robot\tconfig\tresult\tpng\n' >"${manifest}"
fi

camera_args=(
    --camera-look-at 3.40 0.00 1.80 -0.30 0.00 1.10
    --camera-focal-length 2.71
)

# robot:config:backdrop_x_offset
#
# Every entry below is one of the 24 configs from the validated multi-arm sweep,
# used verbatim. The backdrop x-offset exists for split_aloha_fl alone: that
# platform stands on the FLOOR beside the bench and reaches over its edge, so
# lab_003's table (x in [-0.735, 0.765]) is slid to line its left edge up with
# lab_001's (x = -0.924) and keep the cup on the bench instead of past its end.
arms=(
    "franka:level1_pick_franka_ref:0.0"
    "fr3:level1_pick_fr3:0.0"
    "piper:level1_pick_piper:0.0"
    "arx_x5:level1_pick_arx_x5:0.0"
    "arx_r5:level1_pick_arx_r5:0.0"
    "widowx_vx300s:level1_pick_widowx:0.0"
    "ur3e_robotiq:level1_pick_ur3e_robotiq:0.0"
    "ur5e_robotiq:level1_pick_ur5e:0.0"
    "ur10e_robotiq:level1_pick_ur10e_robotiq:0.0"
    "xarm6_robotiq:level1_pick_xarm6_robotiq:0.0"
    "xarm7_robotiq:level1_pick_xarm7_robotiq:0.0"
    "gen3_robotiq:level1_pick_gen3_robotiq:0.0"
    "iiwa14_robotiq:level1_pick_iiwa14_robotiq:0.0"
    "doosan_m0609_robotiq:level1_pick_doosan_m0609_robotiq:0.0"
    "doosan_m1013_robotiq:level1_pick_doosan_m1013_robotiq:0.0"
    "fanuc_lrmate200id_robotiq:level1_pick_fanuc_lrmate200id_robotiq:0.0"
    "jaco2:level1_pick_jaco2:0.0"
    "rizon4_robotiq:level1_pick_rizon4_robotiq:0.0"
    "split_aloha_fl:level1_pick_split_aloha_fl:-0.19"
    "abb_gofa_robotiq:level1_pick_abb_gofa_robotiq:0.0"
    "fanuc_crx10ia_robotiq:level1_pick_fanuc_crx10ia_robotiq:0.0"
    "yaskawa_hc10_robotiq:level1_pick_yaskawa_hc10_robotiq:0.0"
    "techman_tm5_900_robotiq:level1_pick_techman_tm5_900_robotiq:0.0"
    "unitree_z1:level1_pick_unitree_z1:0.0"
)

for entry in "${arms[@]}"; do
    IFS=':' read -r robot config bg_dx <<<"${entry}"
    robot_dir="${output_dir}/${robot}"
    log_path="${output_dir}/logs/${robot}.log"

    existing="$(find "${robot_dir}" -maxdepth 1 -type f -name '*_success.png' -print -quit 2>/dev/null)"
    if [[ -n "${existing}" && -s "${existing}" ]]; then
        printf 'SKIP\t%s\talready rendered\n' "${robot}"
        continue
    fi

    mkdir -p "${robot_dir}"
    printf 'START\t%s\t%s\n' "${robot}" "${config}"
    timeout "${timeout_seconds}" "${python_bin}" experiments/main.py \
        --config-dir experiments/robots/config \
        --config-name="${config}" \
        --max-episodes 1 \
        --max-attempts "${max_attempts}" \
        --headless \
        --no-video \
        --renderer RayTracedLighting \
        --render-width "${render_width}" \
        --render-height "${render_height}" \
        --single-camera 0 \
        "${camera_args[@]}" \
        --paper-background-usd assets/chemistry_lab/lab_003/lab_003.usd \
        --paper-background-translation "${bg_dx}" 0.0 0.008 \
        --paper-background-rotation 0.0 0.0 0.0 \
        --paper-background-scale 1.0 \
        --paper-background-room-scale 1.0 \
        --paper-hide-image-planes \
        --paper-hide-prims /World/table /World/Cube \
                           /World/PaperBackdrop/GroundPlane \
                           /World/PaperBackdrop/CylinderLight \
        --paper-lighting \
        --save-png-dir "${robot_dir}" \
        >"${log_path}" 2>&1
    exit_code=$?

    png="$(find "${robot_dir}" -maxdepth 1 -type f -name '*_success.png' -print -quit)"
    if [[ -n "${png}" && -s "${png}" ]]; then
        result="success"
    elif [[ "${exit_code}" -eq 124 ]]; then
        result="timeout"
    else
        result="no_success_frame"
    fi

    printf '%s\t%s\t%s\t%s\n' "${robot}" "${config}" "${result}" "${png}" >>"${manifest}"
    printf 'DONE\t%s\t%s\t%s\n' "${robot}" "${result}" "${png}"
done

printf '\nGallery: %s\n' "${output_dir}"
column -t -s $'\t' "${manifest}"
