"""Offline seated-pose sensitivity using installed Lula FK and bounded IK.

This changes no robot, config, or scene. Rotation candidates keep the recorded
TCP position fixed, so they are hypotheses, not exact alternate-grasp targets.
No collision or continuous-carry acceptance is claimed.
"""

import argparse
import hashlib
import json
from pathlib import Path

import lula
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--descriptor", type=Path, default=Path("robots/franka/rmpflow/robot_descriptor.yaml"))
    parser.add_argument("--urdf", type=Path, default=Path("robots/franka/lula_franka_gen.urdf"))
    parser.add_argument("--base-position", type=float, nargs=3, required=True)
    parser.add_argument("--yaw-deltas", type=float, nargs="+", default=[-30, -15, -5, 0, 5, 10, 15, 20, 30])
    parser.add_argument("--seed", type=int, default=640134)
    parser.add_argument("--seat-index", type=int, default=0, help="Zero-based seat_multistart record to inspect")
    parser.add_argument("--tcp-local-offset", type=float, nargs=3, default=[0, 0, 0])
    args = parser.parse_args()
    if args.seat_index < 0:
        parser.error("Seat index must be nonnegative")
    with args.trace.open() as trace:
        records = (json.loads(line) for line in trace)
        seats = (row for row in records if row["type"] == "seat_multistart")
        recorded = next((row for i, row in enumerate(seats) if i == args.seat_index), None)
    if recorded is None:
        parser.error("Requested seat record is absent")
    robot = lula.load_robot(str(args.descriptor), str(args.urdf))
    kinematics = robot.kinematics()
    bounds = np.array(
        [[kinematics.c_space_coord_limits(i).lower, kinematics.c_space_coord_limits(i).upper] for i in range(7)]
    )
    lower, upper = bounds[:, 0] + 0.1, bounds[:, 1] - 0.1
    position = np.asarray(recorded["target"]) - args.base_position
    orientation = Rotation.from_quat(np.asarray(recorded["orientation"])[[1, 2, 3, 0]])
    seeds = [recorded["current"], *np.random.default_rng(args.seed).uniform(lower, upper, (8, 7))]
    results = []
    for delta in args.yaw_deltas:
        target = Rotation.from_euler("z", delta, degrees=True) * orientation
        fk_position = position - target.apply(args.tcp_local_offset)

        def residual(q, target=target, fk_position=fk_position):
            pose = kinematics.pose(q, "right_gripper")
            angular = (Rotation.from_matrix(pose.rotation.matrix()) * target.inv()).as_rotvec()
            return np.r_[np.asarray(pose.translation).reshape(3) - fk_position, angular]

        best = None
        for seed in seeds:
            result = least_squares(
                residual,
                np.clip(seed, lower + 1e-9, upper - 1e-9),
                bounds=(lower, upper),
                max_nfev=200,
                ftol=1e-10,
                xtol=1e-10,
                gtol=1e-10,
            )
            error = residual(result.x)
            record = {
                "yaw_delta_deg": delta,
                "position_error": float(np.linalg.norm(error[:3])),
                "angle_error": float(np.linalg.norm(error[3:])),
                "joints": result.x.tolist(),
                "joint_margin": float(np.min(np.minimum(result.x - bounds[:, 0], bounds[:, 1] - result.x))),
            }
            if best is None or np.linalg.norm(error) < best[0]:
                best = (np.linalg.norm(error), record)
        record = best[1]
        record["pose_feasible"] = record["position_error"] <= 0.0005 and record["angle_error"] <= 0.005
        results.append(record)
    print(
        json.dumps(
            {
                "diagnostic_only": True,
                "collision_checked": False,
                "continuous_carry_checked": False,
                "alternate_grasp_position_recomputed": False,
                "assumed_base_orientation_wxyz": [1, 0, 0, 0],
                "base_position": args.base_position,
                "seat_index": args.seat_index,
                "tcp_local_offset": args.tcp_local_offset,
                "results": results,
                "source_sha256": {
                    str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in [Path(__file__), args.trace, args.descriptor, args.urdf]
                },
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
