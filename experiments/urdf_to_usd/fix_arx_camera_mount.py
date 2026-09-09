"""Re-aim the ARX X5 / R5 wrist-camera mount.

The bug
-------
``arx_{x5,r5}.urdf`` mounts the ``camera`` link with ``rpy="-2.093 0 -1.5708"``.
That is a ROS *optical frame*: it puts the link's +Z along the viewing direction,
which lands at (0.867, 0, -0.499) -- forward down the approach axis, 29.92 deg
below horizontal, i.e. squarely on the gripper. Correct, for a sensor frame.

But this link carries no sensor. It carries the camera module's VISUAL mesh, and
that mesh is authored lens-forward along its own +X: in link-local coordinates it
is a 23 mm slab spanning x in [-0.0231, 0] with the lens face flush at x = 0.
Applying an optical-frame rotation to it swings +X onto (0, -1, 0) -- the module
renders pointing straight out the side of the wrist, 90 deg away from what it is
supposed to see.

The fix
-------
Aim the mesh's own axis where the optical axis was aiming. Since the authored
roll is exactly -(pi/2 + pitch), the intended down-tilt falls out as
``pitch = -roll - pi/2 = 0.5222 rad``, and ``rpy = (0, 0.5222, 0)`` sends the
lens to the identical (0.867, 0, -0.499). Nothing else moves: the link is a
kinematic leaf behind a fixed joint, it is not the IK frame (that is
``gripper_center``), and RMPFlow takes its collision spheres from the rmpflow
YAML rather than from this mesh. So no tuned grasp is affected.

Why patch the USD instead of re-importing
-----------------------------------------
Re-running the full URDF->USD import would regenerate collision approximations
and mass properties that the validated 24-arm pick sweep depends on. This edits
only the two places the rotation actually lives -- the fixed joint's
``physics:localRot0`` (what PhysX drives the body to at runtime) and the link's
authored ``xformOp:orient`` -- and leaves everything else byte-identical. The
URDFs are corrected too, so a future re-import produces the same result.

Usage:
    python -m experiments.urdf_to_usd.fix_arx_camera_mount --check   # report only
    python -m experiments.urdf_to_usd.fix_arx_camera_mount
"""

import argparse
import math

from loguru import logger
from pxr import Gf, Usd, UsdGeom

TARGETS = [
    ("experiments/robots/assets/robots/arx_x5.usd", "experiments/robots/definitions/arx_x5/arx_x5.urdf"),
    ("experiments/robots/assets/robots/arx_r5.usd", "experiments/robots/definitions/arx_r5/arx_r5.urdf"),
]
CAMERA_PRIM = "/ARX_5/camera"
CAMERA_JOINT = "/ARX_5/joints/camera_fixed"
# The authored optical-frame roll, as it appears in both URDFs.
ORIGINAL_ROLL_RAD = -2.093
ORIGINAL_RPY_TEXT = 'rpy="-2.093 0 -1.5708"'


def corrected_pitch_rad() -> float:
    """Down-tilt implied by the authored optical roll: roll = -(pi/2 + pitch)."""
    return -ORIGINAL_ROLL_RAD - math.pi / 2.0


def corrected_quat() -> Gf.Quatd:
    """Ry(pitch) as a USD (scalar-first) quaternion."""
    half = corrected_pitch_rad() / 2.0
    return Gf.Quatd(math.cos(half), Gf.Vec3d(0.0, math.sin(half), 0.0))


def patch_usd(usd_path: str, quat: Gf.Quatd, check_only: bool) -> bool:
    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        logger.error(f"Could not open {usd_path}")
        return False

    camera = stage.GetPrimAtPath(CAMERA_PRIM)
    joint = stage.GetPrimAtPath(CAMERA_JOINT)
    if not camera.IsValid() or not joint.IsValid():
        logger.error(f"{usd_path}: expected {CAMERA_PRIM} and {CAMERA_JOINT}")
        return False

    orient_ops = [op for op in UsdGeom.Xformable(camera).GetOrderedXformOps() if op.GetOpName() == "xformOp:orient"]
    if len(orient_ops) != 1:
        logger.error(f"{usd_path}: expected exactly one xformOp:orient on {CAMERA_PRIM}")
        return False
    local_rot = joint.GetAttribute("physics:localRot0")
    if not local_rot or not local_rot.IsValid():
        logger.error(f"{usd_path}: {CAMERA_JOINT} has no physics:localRot0")
        return False

    logger.info(f"{usd_path}")
    logger.info(f"    xformOp:orient      {orient_ops[0].Get()}  ->  {quat}")
    logger.info(f"    physics:localRot0   {local_rot.Get()}  ->  {quat}")
    if check_only:
        return True

    orient_ops[0].Set(quat)
    # localRot0 is a Quatf; assigning the Quatd directly would fail the type check.
    local_rot.Set(Gf.Quatf(quat))
    stage.GetRootLayer().Save()
    return True


def patch_urdf(urdf_path: str, pitch: float, check_only: bool) -> bool:
    with open(urdf_path, encoding="utf-8") as handle:
        text = handle.read()
    if text.count(ORIGINAL_RPY_TEXT) != 1:
        logger.error(f"{urdf_path}: expected exactly one {ORIGINAL_RPY_TEXT}")
        return False
    replacement = f'rpy="0 {pitch:.4f} 0"'
    logger.info(f"{urdf_path}")
    logger.info(f"    camera_fixed origin  {ORIGINAL_RPY_TEXT}  ->  {replacement}")
    if check_only:
        return True
    with open(urdf_path, "w", encoding="utf-8") as handle:
        handle.write(text.replace(ORIGINAL_RPY_TEXT, replacement))
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="report the edits without writing")
    args = parser.parse_args()

    pitch = corrected_pitch_rad()
    quat = corrected_quat()
    logger.info(f"Corrected mount: pitch {pitch:.4f} rad ({math.degrees(pitch):.2f} deg down), quat {quat}")

    ok = True
    for usd_path, urdf_path in TARGETS:
        ok &= patch_usd(usd_path, quat, args.check)
        ok &= patch_urdf(urdf_path, pitch, args.check)

    if not ok:
        logger.error("Some targets could not be patched; nothing further was written")
        return
    logger.success("Checked, nothing written" if args.check else "ARX camera mounts re-aimed")


if __name__ == "__main__":
    main()
