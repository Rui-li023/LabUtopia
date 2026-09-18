"""Read-only guard for calibrated single-actuator Franka finger-force limits.

In the shipped Panda articulation, only joint 1 has a drive; joint 2 follows
through a mimic constraint. The free-coupon probe measures about 17.5 N per
jaw for a 35 N drive cap, not 35 N per jaw. A larger cap is permitted only
with this explicitly verified topology and never above the Hand's published
70 N continuous grasp-force rating. This guard does not author any joint.
"""

import math
from typing import Any


def validate_franka_finger_speed(speed: float) -> None:
    """Franka Hand specifies 50 mm/s travel speed per finger, not 10 m/s."""
    if not math.isfinite(speed) or not 0 < speed <= 0.05:
        raise ValueError("Franka finger velocity limit must be finite and in (0, 0.05] m/s")


def validate_franka_drive_force(force: float, semantics: str) -> None:
    """Keep legacy limits; require explicit single-actuator semantics above 35 N.

    The caller must additionally validate the actual articulation topology
    before using ``single_mimic_drive`` in a simulation.
    """
    if semantics not in ("legacy", "single_mimic_drive"):
        raise ValueError("Franka force semantics must be legacy or single_mimic_drive")
    limit = 70.0 if semantics == "single_mimic_drive" else 35.0
    if not math.isfinite(force) or not 0 < force <= limit:
        raise ValueError(f"Franka {semantics} drive cap must be finite and in (0, {limit:g}] N")


def validate_single_mimic_finger_drive(
    stage: Any, robot_path: str, *, allow_zero_gain_follower: bool = False
) -> dict[str, Any]:
    """Reject absent, independently driven, disabled, or differently geared fingers."""
    paths = [f"{robot_path}/panda_hand/panda_finger_joint{index}" for index in (1, 2)]
    joints = [stage.GetPrimAtPath(path) for path in paths]
    if not all(prim.IsValid() and prim.GetTypeName() == "PhysicsPrismaticJoint" for prim in joints):
        raise ValueError("Single-mimic force requires both Franka prismatic finger joints")
    schemas = [list(prim.GetAppliedSchemas()) for prim in joints]
    drives = [[name for name in names if name.startswith("PhysicsDriveAPI:")] for names in schemas]
    mimics = [[name for name in names if name.startswith("PhysxMimicJointAPI:")] for names in schemas]
    # Some original USDs retain an inert DriveAPI on the mimic. Accepting that
    # topology is explicit; a nonzero follower gain is never a passive hand.
    passive_drive = (
        allow_zero_gain_follower
        and drives[1] == ["PhysicsDriveAPI:linear"]
        and all(
            joints[1].GetAttribute(f"drive:linear:physics:{name}").Get() == 0.0 for name in ("stiffness", "damping")
        )
    )
    if (
        drives != [["PhysicsDriveAPI:linear"], []] and not (drives[0] == ["PhysicsDriveAPI:linear"] and passive_drive)
    ) or mimics != [[], ["PhysxMimicJointAPI:rotX"]]:
        raise ValueError("Single-mimic force requires one linear drive and one undriven mimic follower")
    for joint in joints:
        if (
            joint.GetAttribute("physics:jointEnabled").Get() is not True
            or joint.GetAttribute("physics:excludeFromArticulation").Get() is not False
            or joint.GetAttribute("physics:axis").Get() != "X"
        ):
            raise ValueError("Single-mimic force requires enabled, articulated X-axis finger joints")
    lead, follower = joints
    if lead.GetAttribute("drive:linear:physics:type").Get() != "force":
        raise ValueError("Single-mimic force requires a force-based actuator drive")
    prefix = "physxMimicJoint:rotX:"
    gearing = follower.GetAttribute(prefix + "gearing").Get()
    offset = follower.GetAttribute(prefix + "offset").Get()
    references = [str(path) for path in follower.GetRelationship(prefix + "referenceJoint").GetTargets()]
    if (
        references != [paths[0]]
        or follower.GetAttribute(prefix + "referenceJointAxis").Get() != "rotX"
        or not isinstance(gearing, int | float)
        or not math.isclose(gearing, -1.0, rel_tol=0, abs_tol=1e-6)
        or not isinstance(offset, int | float)
        or not math.isclose(offset, 0.0, rel_tol=0, abs_tol=1e-9)
    ):
        raise ValueError("Single-mimic force requires the calibrated -1 gearing and zero offset")
    return {
        "actuated_joint": paths[0],
        "passive_mimic_joint": paths[1],
        "joint_schemas": schemas,
        "gearing": float(gearing),
        "offset": float(offset),
        "continuous_total_force_cap_N": 70.0,
        "measured_force_not_guaranteed_by_cap": True,
    }
