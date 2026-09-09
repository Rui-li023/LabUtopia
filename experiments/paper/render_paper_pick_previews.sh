#!/usr/bin/env bash

# Render a small approval set of clean, single-view publication frames from the
# same lab_001 pick scene used by the 24-arm benchmark sweep.
set -euo pipefail

repo_dir="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
python_bin="${PYTHON_BIN:-${HOME}/mambaforge/envs/isaacsim5.1/bin/python}"
output_dir="${OUTPUT_DIR:-${repo_dir}/outputs/paper_pick_previews_4k}"
timeout_seconds="${TIMEOUT_SECONDS:-1800}"

if [[ ! -x "${python_bin}" ]]; then
    echo "Isaac Sim Python was not found at ${python_bin}" >&2
    echo "Set PYTHON_BIN to the Isaac Sim 5.1 Python executable." >&2
    exit 2
fi

mkdir -p "${output_dir}"
cd "${repo_dir}"

# The eye/target pair gives a 3/4 view of the arm, work surface, and cabinets.
# Keep this fixed across arms so the approval frames remain comparable.
camera_args=(
    --camera-look-at 4.20 -5.80 3.30 0.00 0.00 1.00
    --camera-focal-length 12.0
)

# Use the authored lab_003 environment as the publication backdrop. It is kept
# as a separate reference so the lab_001 pick layout and robot calibration stay
# unchanged.
background_args=(
    --paper-background-usd assets/chemistry_lab/lab_003/lab_003.usd
    --paper-background-translation 0.0 6.0 0.0
    --paper-background-rotation 0.0 0.0 180.0
    --paper-background-scale 1.0
    --paper-background-room-scale 1.0
    --paper-hide-image-planes
)

arms=(
    "franka:level1_pick_franka_ref"
    "piper:level1_pick_piper"
    "ur5e_robotiq:level1_pick_ur5e"
    "gen3_robotiq:level1_pick_gen3_robotiq"
)

for entry in "${arms[@]}"; do
    robot="${entry%%:*}"
    config="${entry#*:}"
    robot_dir="${output_dir}/${robot}"
    mkdir -p "${robot_dir}"
    echo "Rendering ${robot} (${config}) -> ${robot_dir}"
    timeout "${timeout_seconds}" "${python_bin}" experiments/main.py \
        --config-dir experiments/robots/config \
        --config-name="${config}" \
        --max-episodes 1 \
        --max-attempts 1 \
        --headless \
        --no-video \
        --renderer RayTracedLighting \
        --render-width 3840 \
        --render-height 2160 \
        --single-camera 0 \
        "${camera_args[@]}" \
        "${background_args[@]}" \
        --paper-lighting \
        --save-png-dir "${robot_dir}" \
        >"${robot_dir}/render.log" 2>&1
    png="$(find "${robot_dir}" -maxdepth 1 -type f -name 'episode_*_success.png' -print -quit)"
    if [[ -z "${png}" || ! -s "${png}" ]]; then
        echo "${robot}: expected successful PNG was not produced; see ${robot_dir}/render.log" >&2
        exit 1
    fi
    echo "Saved ${png}"
done

echo "Approval renders are in ${output_dir}"
