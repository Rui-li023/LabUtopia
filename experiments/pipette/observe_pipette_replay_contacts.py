"""Observe a bounded replay contact window without changing recorded actions.

The ordinary grasp observer still runs main.py and BaseController replay.
This adds contact reporting (not contact parameters), joint observations, and
the exact action clock to diagnose collection/replay contact divergence.
An extra diagnostic replay never counts as an independent acceptance sample.
"""

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path

import numpy as np


def contact_records(headers, contacts, decode_path, *, step, roots, threshold):
    """Serialize selected pairs using each header's own contact-data slice."""
    for header in headers:
        paths = [str(decode_path(value)) for value in (header.collider0, header.collider1)]
        if not any(path == root or path.startswith(root + "/") for path in paths for root in roots):
            continue
        first = int(header.contact_data_offset)
        count = int(header.num_contact_data)
        if first < 0 or count < 0 or first + count > len(contacts):
            raise ValueError("Contact header points outside the supplied contact array")
        for contact in contacts[first : first + count]:
            impulse = np.array([contact.impulse.x, contact.impulse.y, contact.impulse.z], dtype=float)
            if not np.isfinite(impulse).all():
                raise ValueError("Non-finite contact impulse")
            if np.linalg.norm(impulse) <= threshold:
                continue
            yield {
                "type": "contact",
                "replay_action_step": step,
                "paths": paths,
                "separation_m": float(contact.separation),
                "position_world": [float(getattr(contact.position, axis)) for axis in "xyz"],
                "normal_world": [float(getattr(contact.normal, axis)) for axis in "xyz"],
                "impulse_Ns": impulse.tolist(),
            }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contacts", type=Path, required=True)
    parser.add_argument("--grip", type=Path, required=True)
    parser.add_argument("--through-step", type=int, default=3500)
    parser.add_argument("--tool-root", action="append", default=None)
    parser.add_argument("main_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    main_args = args.main_args[1:] if args.main_args[:1] == ["--"] else args.main_args
    if not main_args or args.through_step < 1:
        parser.error("Positive contact window and main.py arguments are required")
    if args.contacts.resolve() == args.grip.resolve() or args.contacts.exists() or args.grip.exists():
        parser.error("Contact and grip outputs must be distinct new files")
    roots = tuple(args.tool_root or ["/World/pipette_01"])
    if any(not root.startswith("/World/") or root.endswith("/") for root in roots):
        parser.error("Use explicit tool prim roots under /World, without a trailing slash")
    args.contacts.parent.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    original_argv = sys.argv
    sys.argv = [str(root / "experiments/main.py"), *main_args]
    # SimulationApp must start before Isaac/PhysX imports. No controller or
    # task implementation is replaced; all wrappers delegate exact results.
    simulation = importlib.import_module("experiments.main")
    usd = importlib.import_module("tasks.pipette_rack_task")
    controller_type = importlib.import_module("controllers.pipette_rack_controller").PipetteRackTaskController
    observer = importlib.import_module("experiments.pipette.observe_pipette_grip")
    original_create = simulation.create_task
    original_step = controller_type.step
    own_step = controller_type.__dict__.get("step")
    clock = {"step": None}
    with args.contacts.open("x", encoding="utf-8") as stream:

        def emit(record):
            stream.write(json.dumps(record, allow_nan=False) + "\n")

        emit(
            {
                "type": "metadata",
                "independent_sample": False,
                "read_only_actions": True,
                "contact_reporting_only": True,
                "through_step": args.through_step,
                "tool_roots": roots,
                "impulse_threshold_Ns": 0.001,
                "argv": main_args,
                "source_sha256": {
                    str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in (Path(__file__), root / "experiments/pipette/observe_pipette_grip.py", root / "experiments/main.py")
                },
            }
        )

        def create_task(*create_args, **create_kwargs):
            task = original_create(*create_args, **create_kwargs)
            if str(task.cfg.mode) != "replay":
                raise ValueError("This contact observer is replay-only")
            bodies = []
            for tool_root in roots:
                prim = task.stage.GetPrimAtPath(tool_root)
                if not prim.IsValid():
                    raise ValueError(f"Missing observed tool: {tool_root}")
                for body in usd.Usd.PrimRange(prim):
                    if body.HasAPI(usd.UsdPhysics.RigidBodyAPI):
                        usd.PhysxSchema.PhysxContactReportAPI.Apply(body).CreateThresholdAttr(0.0)
                        bodies.append(str(body.GetPath()))

            def report(headers, contacts):
                if clock["step"] is None or clock["step"] > args.through_step:
                    return
                for record in contact_records(
                    headers,
                    contacts,
                    usd.PhysicsSchemaTools.intToSdfPath,
                    step=clock["step"],
                    roots=roots,
                    threshold=0.001,
                ):
                    emit(record)

            task._replay_contact_observation = usd.get_physx_simulation_interface().subscribe_contact_report_events(
                report
            )
            emit({"type": "contact_reporting_setup", "rigid_bodies": bodies})
            return task

        def step(active, state):
            index = int(active._current_action_step)
            if index <= args.through_step:
                emit(
                    {
                        "type": "joint_sample",
                        "replay_action_step": index,
                        "joint_positions": np.asarray(state["joint_positions"]).tolist(),
                        "joint_velocities": np.asarray(active.robot.get_joint_velocities()).tolist(),
                    }
                )
            result = original_step(active, state)
            clock["step"] = int(active._current_action_step)
            if result[1]:
                emit({"type": "terminal", "success": bool(result[2]), "replay_action_step": clock["step"]})
                stream.flush()
            return result

        simulation.create_task = create_task
        controller_type.step = step
        sys.argv = [str(root / "experiments/pipette/observe_pipette_grip.py"), "--output", str(args.grip), "--", *main_args]
        try:
            observer.main()
        finally:
            simulation.create_task = original_create
            if own_step is None:
                del controller_type.step
            else:
                controller_type.step = own_step
            sys.argv = original_argv


if __name__ == "__main__":
    main()
