"""Probe a robot articulation's idle stability in a minimal Isaac Sim world.

This is intentionally separate from the asset verifier: a drive can respond to a
short command and still become unstable once gravity, joint limits, mimic joints,
and position targets interact over several physics steps.

Example::

    python scripts/urdf_to_usd/diagnose_articulation.py \
        --usd assets/robots/fr3.usd \
        --positions 0,-0.785398,0,-2.356194,0,1.570796,0.785398,0.04,0.04 \
        --damping-ratio 0.02 --steps 300
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from isaacsim import SimulationApp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--usd", required=True)
    parser.add_argument(
        "--positions",
        help="Comma-separated initial joint positions in articulation DOF order",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--damping-ratio",
        type=float,
        default=None,
        help="Raise each driven joint's kd to at least kp times this ratio",
    )
    parser.add_argument(
        "--passive-joint",
        action="append",
        default=[],
        help="Joint name whose position drive should be disabled; repeat as needed",
    )
    parser.add_argument("--disable-gravity", action="store_true")
    return parser.parse_args()


ARGS = parse_args()
APP = SimulationApp({"headless": True})

from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import SingleArticulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from isaacsim.core.utils.types import ArticulationAction  # noqa: E402


def as_list(values: np.ndarray | None) -> list[float] | None:
    if values is None:
        return None
    return [round(float(value), 7) for value in np.asarray(values).reshape(-1)]


def report(articulation: SingleArticulation, label: str) -> None:
    positions = articulation.get_joint_positions()
    velocities = articulation.get_joint_velocities()
    max_position = float(np.nanmax(np.abs(positions)))
    max_velocity = float(np.nanmax(np.abs(velocities)))
    print(
        f"[state] {label} q={as_list(positions)} qd={as_list(velocities)} "
        f"max_abs_q={max_position:.7g} max_abs_qd={max_velocity:.7g}",
        flush=True,
    )


def main() -> None:
    usd_path = os.path.abspath(ARGS.usd)
    world = World(stage_units_in_meters=1.0)
    add_reference_to_stage(usd_path=usd_path, prim_path="/World/Robot")
    articulation = SingleArticulation(prim_path="/World/Robot", name="robot_probe")
    world.scene.add(articulation)
    world.reset()

    names = list(articulation.dof_names)
    print(f"[asset] usd={usd_path} dof={articulation.num_dof} names={names}", flush=True)
    report(articulation, "after_reset")

    controller = articulation.get_articulation_controller()
    stiffness, damping = controller.get_gains()
    stiffness = np.asarray(stiffness, dtype=float)
    damping = np.asarray(damping, dtype=float)
    print(f"[drives] kp={as_list(stiffness)} kd={as_list(damping)}", flush=True)

    view = articulation._articulation_view
    for method_name in ("get_max_efforts", "get_max_joint_velocities"):
        method = getattr(view, method_name, None)
        if method is None:
            continue
        try:
            print(f"[limits] {method_name}={as_list(method())}", flush=True)
        except Exception as exc:
            print(f"[limits] {method_name}=unavailable ({exc})", flush=True)

    if ARGS.positions:
        positions = np.asarray([float(value) for value in ARGS.positions.split(",")], dtype=np.float32)
        if len(positions) != articulation.num_dof:
            raise ValueError(
                f"--positions has {len(positions)} values; articulation has {articulation.num_dof} DOFs: {names}"
            )
    else:
        positions = articulation.get_joint_positions().copy()

    if ARGS.damping_ratio is not None:
        damping = np.maximum(damping, stiffness * ARGS.damping_ratio)

    for joint_name in ARGS.passive_joint:
        if joint_name not in names:
            raise ValueError(f"Unknown passive joint {joint_name!r}; available: {names}")
        index = names.index(joint_name)
        stiffness[index] = 0.0
        damping[index] = 0.0

    controller.set_gains(kps=stiffness, kds=damping)
    articulation.set_joints_default_state(
        positions=positions,
        velocities=np.zeros(articulation.num_dof, dtype=np.float32),
    )
    articulation.set_joint_positions(positions)
    articulation.set_joint_velocities(np.zeros(articulation.num_dof, dtype=np.float32))
    articulation.apply_action(ArticulationAction(joint_positions=positions))
    if ARGS.disable_gravity:
        articulation.disable_gravity()
    report(articulation, "initialized")
    new_stiffness, new_damping = controller.get_gains()
    print(
        f"[drives] applied_kp={as_list(new_stiffness)} applied_kd={as_list(new_damping)}",
        flush=True,
    )

    checkpoints = {1, 2, 5, 10, 30, 60, 120, 180, 240, ARGS.steps}
    for step in range(1, ARGS.steps + 1):
        world.step(render=False)
        if step in checkpoints:
            report(articulation, f"step={step}")
        positions_now = articulation.get_joint_positions()
        velocities_now = articulation.get_joint_velocities()
        if (
            not np.all(np.isfinite(positions_now))
            or not np.all(np.isfinite(velocities_now))
            or np.max(np.abs(positions_now)) > 100.0
            or np.max(np.abs(velocities_now)) > 1000.0
        ):
            print(f"[result] UNSTABLE at step={step}", flush=True)
            break
    else:
        drift = np.max(np.abs(articulation.get_joint_positions() - positions))
        print(f"[result] STABLE max_position_drift={float(drift):.7g}", flush=True)

    world.stop()
    APP.close()


if __name__ == "__main__":
    main()
