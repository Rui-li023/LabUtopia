#!/usr/bin/env bash

set -uo pipefail

repo_dir="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
run_dir="${RUN_DIR:-${repo_dir}/test_outputs/all_arm_pick_$(date +%Y%m%d)}"
python_bin="${PYTHON_BIN:-python}"
timeout_seconds="${TIMEOUT_SECONDS:-900}"

arms=(
    "franka:level1_pick"
    "fr3:level1_pick_fr3"
    "piper:level1_pick_piper"
    "arx_x5:level1_pick_arx_x5"
    "arx_r5:level1_pick_arx_r5"
    "widowx_vx300s:level1_pick_widowx"
    "ur3e_robotiq:level1_pick_ur3e_robotiq"
    "ur5e_robotiq:level1_pick_ur5e"
    "ur10e_robotiq:level1_pick_ur10e_robotiq"
    "xarm6_robotiq:level1_pick_xarm6_robotiq"
    "xarm7_robotiq:level1_pick_xarm7_robotiq"
    "gen3_robotiq:level1_pick_gen3_robotiq"
    "iiwa14_robotiq:level1_pick_iiwa14_robotiq"
    "doosan_m0609_robotiq:level1_pick_doosan_m0609_robotiq"
    "doosan_m1013_robotiq:level1_pick_doosan_m1013_robotiq"
    "fanuc_lrmate200id_robotiq:level1_pick_fanuc_lrmate200id_robotiq"
    "jaco2:level1_pick_jaco2"
    "rizon4_robotiq:level1_pick_rizon4_robotiq"
    "split_aloha_fl:level1_pick_split_aloha_fl"
    "abb_gofa_robotiq:level1_pick_abb_gofa_robotiq"
    "fanuc_crx10ia_robotiq:level1_pick_fanuc_crx10ia_robotiq"
    "yaskawa_hc10_robotiq:level1_pick_yaskawa_hc10_robotiq"
    "techman_tm5_900_robotiq:level1_pick_techman_tm5_900_robotiq"
    "unitree_z1:level1_pick_unitree_z1"
)

mkdir -p "${run_dir}/logs"
manifest="${run_dir}/manifest.tsv"
if [[ ! -f "${manifest}" ]]; then
    printf 'robot\tconfig\texit_code\tresult\tvideo\n' >"${manifest}"
fi

cd "${repo_dir}" || exit 1
env_args=()
if [[ -n "${PYTHON_LIBS:-}" ]]; then
    env_args=(env "LD_LIBRARY_PATH=${PYTHON_LIBS}")
fi

for entry in "${arms[@]}"; do
    robot="${entry%%:*}"
    config="${entry#*:}"
    log_path="${run_dir}/logs/${robot}.log"

    if [[ -f "${log_path}" ]] && grep -q 'Video saved:' "${log_path}"; then
        video="$(sed -E $'s/\033\[[0-9;]*[mK]//g' "${log_path}" | sed -n 's/.*Video saved: //p' | tail -n 1)"
        result="${video##*_}"
        result="${result%.mp4}"
        if ! grep -q "^${robot}"$'\t' "${manifest}"; then
            printf '%s\t%s\t0\t%s\t%s\n' \
                "${robot}" "${config}" "${result}" "${video}" >>"${manifest}"
        fi
        printf 'SKIP\t%s\talready rendered\n' "${robot}"
        continue
    fi

    printf 'START\t%s\t%s\n' "${robot}" "${config}"
    timeout "${timeout_seconds}" "${env_args[@]}" \
        "${python_bin}" main.py \
        --config-name="${config}" \
        --max-episodes 1 \
        --max-attempts 1 \
        --headless >"${log_path}" 2>&1
    exit_code=$?

    video="$(sed -E $'s/\033\[[0-9;]*[mK]//g' "${log_path}" | sed -n 's/.*Video saved: //p' | tail -n 1)"

    if [[ -n "${video}" && -f "${video}" ]]; then
        result="${video##*_}"
        result="${result%.mp4}"
    elif [[ "${exit_code}" -eq 124 ]]; then
        result="timeout"
    else
        result="error"
    fi

    printf '%s\t%s\t%s\t%s\t%s\n' \
        "${robot}" "${config}" "${exit_code}" "${result}" "${video}" >>"${manifest}"
    printf 'DONE\t%s\texit=%s\tresult=%s\t%s\n' \
        "${robot}" "${exit_code}" "${result}" "${video}"
done
