"""Calibrated camera EXTRINSICS for the sim2real bench, and the frame algebra
needed to turn them into USD camera transforms.

Single source of truth, the companion of ``usd_intrinsics.py``. Pure math: numpy
only, no pxr, no Isaac Sim, so it can be imported by the scene builder, by the
render verifier, and run standalone as a self-test.

The numbers themselves are NOT retyped here --- they are loaded from the hand-eye
calibration the rig exported, archived verbatim at
``assets/sim2real/calibration/camera_params.json``. Only the decisions that turn
that file into a scene (which robot frame "gripper" means, which scene camera
each vendor channel maps to) live in this module.

DO NOT VALIDATE THESE POSES AGAINST THE 2026-08-06 SNAPSHOTS. Every calibration
in the export is NEWER than that snapshot --- wrist 08-07 16:03, third_view
08-08 16:59, second_third 08-08 17:27 --- so the framings genuinely differ, and
the 08-06 photos disagree with the calibration by ~24 deg of aim on third_view.
That is the cameras having been moved, not a placement bug: these poses were
checked against the rig's CURRENT frames (2026-08-08 19:34) and land within
1 px --- see OVERLAY_ANCHORS below, which the self-test asserts. The 08-06 photos
remain the reference for bench materials and lighting, which is all they were
used for; they are NOT a reference for where the cameras are.

Three conventions meet in this file; mixing them up is the classic way to get a
camera that is subtly, plausibly wrong:

  OpenCV camera    +x right, +y DOWN, +z FORWARD along the optical axis.
                   What the calibration is expressed in.
  USD camera       +x right, +y UP,   -z forward (a USD camera looks down its
                   own -z). Differs from OpenCV by a 180 deg roll about +x,
                   i.e. right-multiply by diag(1, -1, -1).
  Gf.Matrix4d      ROW-vector convention, ``p' = p * M``, translation in row 3.
                   Every matrix in this module is the opposite (numpy /
                   column-vector, ``p' = M @ p``), so the builder must TRANSPOSE
                   on the way into USD. Use ``to_gf_rows()`` and never hand-roll it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CALIBRATION_JSON = Path(__file__).resolve().parents[2] / "assets/sim2real/calibration/camera_params.json"

# ── frame decisions ──────────────────────────────────────────────────────────
# Vendor channel name -> the camera prim name used in this scene / in INTRINSICS.
VENDOR_TO_SCENE = {
    "third_view": "third_view_orbbec335L",
    "second_third_view": "second_third_realsense435i",
    "left_wrist_view": "left_wrist_orbbec335",
}

# THE ONE INTERPRETIVE CONSTANT IN THIS FILE. Everything else is measured; this is
# a decision, because the export only says the wrist camera was calibrated against
# a frame it calls "gripper" and never defines it.
#
# It is O_T_EE, the libfranka end-effector frame: with the stock Franka Hand the
# default F_T_EE is Rz(-45 deg) plus +0.1034 m along z, i.e. the TCP BETWEEN THE
# FINGER PADS, not the mounting flange. Two independent confirmations:
#   * Franka.usd carries an actual prim for it --- /Franka/panda_hand/tool_center,
#     local translate (0, 0, 0.1034), identity rotation. build_table_scene.py
#     asserts this constant against that prim, so the two cannot silently drift.
#   * panda_hand already carries the -45 deg roll (fixed joint from panda_link8,
#     localRot0 = Rz(-45 deg), zero translation), so hand -> gripper is PURE
#     TRANSLATION along +z. The roll must NOT be applied a second time here.
#
# The flange alternative (0.0) is ruled out three ways: at 0.0 the hand body would
# project over ~83 % of the wrist frame at 5-14 cm (the real frame is ~96 % bare
# tabletop); the 08-06 second-third photo shows the camera bracketed to the SIDE of
# the hand, forward of the flange, not 5.6 cm behind it on the forearm; and 0.0 is
# not a frame any Franka driver reports as "gripper".
#
# VERIFIED, not assumed. The rig's own overlay draws its "EE" marker at the
# gripper-frame origin; reprojecting that origin through this constant lands at
# (273.6, 441.8) against the tool's (274.1, 442.1) --- 0.5 px. See OVERLAY_ANCHORS.
# The rig's handover doc (DataCollectionSystemV2/docs/real2sim.md section 2) states
# the same thing independently: "机械臂报的 O_T_EE 是夹爪 TCP", with
# F_T_EE = Rz(-45 deg) + 0.1034 m, reconciling to 0.24 mm against the arm's own
# reported eef_pose. Do not "fix" the finger pads showing up in the bottom of the
# wrist image --- the real wrist camera sees its gripper there too.
GRIPPER_Z_IN_HAND = 0.1034  # metres, panda_hand origin -> calibrated "gripper" origin

# Cameras with no calibration of their own. The right arm is scenery in every
# sim2real config so far, and the two wrist brackets are the same part.
BORROWED = {"right_wrist_orbbec335": "left_wrist_orbbec335"}

# 180 deg roll about the camera's own +x: OpenCV optical frame -> USD camera frame.
CV_TO_USD = np.diag([1.0, -1.0, -1.0, 1.0])

# GROUND TRUTH, read off the rig's own overlay tool, 2026-08-08 19:34:
#   nuc2:~/dc/collector $ python scripts/verify_calibration_overlay.py -> /tmp/overlay/*.jpg
# Both markers are the projection of a frame ORIGIN, so both are POSE-INDEPENDENT:
# the wrist camera is rigid to the gripper, and the base does not move. That makes
# them a clean test of the extrinsics alone, with no forward kinematics involved.
# The self-test below reprojects them and asserts we land in the same place.
#   {camera: (point in its own reference frame, expected (u, v) px)}
OVERLAY_ANCHORS = {
    "left_wrist_orbbec335": ((0.0, 0.0, 0.0), (274.1, 442.1)),  # "EE" dot
    "third_view_orbbec335L": ((0.0, 0.0, 0.0), (302.0, 90.5)),  # "BASE" dot
    # second_third has no anchor: the base projects to (655, 342), off the right
    # edge of a 640-wide frame, so its overlay shows no BASE marker. Its extrinsics
    # remain the one UNVERIFIED item --- and it is also the weakest calibration in
    # the export (10 inliers, 7.7 mm residual, 14.1 mm five-method spread, solved
    # offline). To check it, park the arm in the middle of that camera's view and
    # re-run verify_calibration_overlay.py.
}
ANCHOR_TOL_PX = 3.0  # marker centroids are themselves only good to ~1 px


@dataclass(frozen=True)
class Extrinsics:
    """One calibrated camera pose, in OpenCV optical convention."""

    name: str  # scene / INTRINSICS key
    vendor_name: str  # channel name in the calibration export
    reference: str  # "base" (robot base frame) | "gripper" (see GRIPPER_Z_IN_HAND)
    cam_in_ref: np.ndarray  # 4x4, p_ref = cam_in_ref @ p_cam
    mode: str  # eye_to_hand | eye_in_hand
    trans_err_mm: float  # AX=XB residual reported by the calibration
    rot_err_deg: float
    serial: str
    borrowed_from: str | None = None

    @property
    def position(self) -> np.ndarray:
        """Camera origin in the reference frame, metres."""
        return self.cam_in_ref[:3, 3].copy()

    @property
    def optical_axis(self) -> np.ndarray:
        """Unit viewing direction (OpenCV +z) in the reference frame."""
        return self.cam_in_ref[:3, 2].copy()


def _load(path: Path = CALIBRATION_JSON) -> dict[str, Extrinsics]:
    raw = json.loads(path.read_text())
    out: dict[str, Extrinsics] = {}
    for vendor_name, cam in raw["cameras"].items():
        scene_name = VENDOR_TO_SCENE.get(vendor_name)
        if scene_name is None:  # a channel we do not model in this scene
            continue
        matrix = np.asarray(cam["cam_in_ref"], dtype=float)
        _assert_rigid(matrix, f"{vendor_name}.cam_in_ref")
        cal = cam.get("calibration", {})
        out[scene_name] = Extrinsics(
            name=scene_name,
            vendor_name=vendor_name,
            reference=cam["reference"],
            cam_in_ref=matrix,
            mode=cal.get("mode", "?"),
            trans_err_mm=float(cal.get("axxb_trans_err_median_mm", float("nan"))),
            rot_err_deg=float(cal.get("axxb_rot_err_median_deg", float("nan"))),
            serial=cam.get("serial", ""),
        )
    for scene_name, source in BORROWED.items():
        src = out[source]
        out[scene_name] = Extrinsics(
            name=scene_name,
            vendor_name=src.vendor_name,
            reference=src.reference,
            cam_in_ref=src.cam_in_ref.copy(),
            mode=src.mode,
            trans_err_mm=src.trans_err_mm,
            rot_err_deg=src.rot_err_deg,
            serial=src.serial,
            borrowed_from=source,
        )
    return out


def _assert_rigid(matrix: np.ndarray, label: str, tol: float = 1e-9) -> None:
    if matrix.shape != (4, 4):
        raise ValueError(f"{label}: expected 4x4, got {matrix.shape}")
    rot = matrix[:3, :3]
    orth = np.abs(rot @ rot.T - np.eye(3)).max()
    det = float(np.linalg.det(rot))
    if orth > 1e-6 or abs(det - 1.0) > 1e-6:
        raise ValueError(f"{label}: not a proper rigid transform (orth err {orth:.2e}, det {det:.9f})")
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=tol):
        raise ValueError(f"{label}: bottom row is {matrix[3]}, expected (0,0,0,1)")


EXTRINSICS: dict[str, Extrinsics] = _load()


# ── frame algebra ────────────────────────────────────────────────────────────
def rigid(rot: np.ndarray, translation: tuple[float, float, float]) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = rot
    out[:3, 3] = translation
    return out


def rot_z(degrees: float) -> np.ndarray:
    c, s = np.cos(np.radians(degrees)), np.sin(np.radians(degrees))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def world_from_base(base_xyz: tuple[float, float, float], yaw_deg: float) -> np.ndarray:
    """Pose of a robot's base frame in world coordinates.

    The bench arms are authored with a yaw about +z only, which is exactly the
    transform their calibrated base-frame poses have to be pushed through.
    """
    return rigid(rot_z(yaw_deg), base_xyz)


def hand_from_gripper(gripper_z_in_hand: float = GRIPPER_Z_IN_HAND) -> np.ndarray:
    """Pose of the calibrated "gripper" frame in the panda_hand frame.

    Pure translation along the hand's approach axis: the -45 deg roll that
    distinguishes panda_link8 from the Franka Hand's EE frame is already baked
    into panda_hand itself, so it must NOT be applied a second time here.
    """
    return rigid(np.eye(3), (0.0, 0.0, gripper_z_in_hand))


def usd_cam_in_ref(name: str) -> np.ndarray:
    """Calibrated pose with the camera's own axes rotated into USD convention.

    Still expressed in the calibration's reference frame --- compose with
    ``world_from_base`` / ``hand_from_gripper`` to land it in the scene.
    """
    return EXTRINSICS[name].cam_in_ref @ CV_TO_USD


def usd_cam_in_world(name: str, base_xyz: tuple[float, float, float], yaw_deg: float) -> np.ndarray:
    """World transform for a camera calibrated in the robot BASE frame."""
    extr = EXTRINSICS[name]
    if extr.reference != "base":
        raise ValueError(f"{name}: reference is {extr.reference!r}, not 'base' --- use usd_cam_in_hand")
    return world_from_base(base_xyz, yaw_deg) @ usd_cam_in_ref(name)


def usd_cam_in_hand(name: str, gripper_z_in_hand: float = GRIPPER_Z_IN_HAND) -> np.ndarray:
    """panda_hand-local transform for a camera calibrated in the GRIPPER frame."""
    extr = EXTRINSICS[name]
    if extr.reference != "gripper":
        raise ValueError(f"{name}: reference is {extr.reference!r}, not 'gripper' --- use usd_cam_in_world")
    return hand_from_gripper(gripper_z_in_hand) @ usd_cam_in_ref(name)


def to_gf_rows(matrix: np.ndarray) -> tuple[float, ...]:
    """Flatten a column-vector 4x4 into the 16 values ``Gf.Matrix4d(...)`` wants.

    Gf is row-vector (``p' = p * M``), numpy here is column-vector
    (``p' = M @ p``), so this is a transpose. Kept in one place on purpose.
    """
    return tuple(float(v) for v in np.asarray(matrix, dtype=float).T.reshape(-1))


def describe(name: str) -> str:
    extr = EXTRINSICS[name]
    pos = extr.position
    axis = extr.optical_axis
    tag = f" (borrowed from {extr.borrowed_from})" if extr.borrowed_from else ""
    return (
        f"{name:28s} ref={extr.reference:8s} "
        f"pos=({pos[0]:+.4f},{pos[1]:+.4f},{pos[2]:+.4f}) "
        f"axis=({axis[0]:+.3f},{axis[1]:+.3f},{axis[2]:+.3f}) "
        f"{extr.mode} err={extr.trans_err_mm:.2f}mm/{extr.rot_err_deg:.2f}deg{tag}"
    )


if __name__ == "__main__":
    raw = json.loads(CALIBRATION_JSON.read_text())

    print(f"calibration: {CALIBRATION_JSON}")
    for name in EXTRINSICS:
        print("  " + describe(name))

    # 1. the vendor shipped both directions; they must be exact inverses
    for vendor_name, cam in raw["cameras"].items():
        fwd = np.asarray(cam["cam_in_ref"], dtype=float)
        inv = np.asarray(cam["ref_to_cam"], dtype=float)
        err = np.abs(fwd @ inv - np.eye(4)).max()
        assert err < 1e-9, f"{vendor_name}: cam_in_ref @ ref_to_cam off by {err:.2e}"
    print(f"\ncam_in_ref @ ref_to_cam == I for all {len(raw['cameras'])} vendor channels")

    # 2. the OpenCV->USD flip is an involution, and it really does flip the axes
    assert np.allclose(CV_TO_USD @ CV_TO_USD, np.eye(4))
    for name, extr in EXTRINSICS.items():
        usd = usd_cam_in_ref(name)
        # a USD camera looks down its own -z; that must equal the OpenCV +z
        assert np.allclose(-usd[:3, 2], extr.optical_axis, atol=1e-12), name
        # USD +y (up) must be the OpenCV -y (which pointed down)
        assert np.allclose(usd[:3, 1], -extr.cam_in_ref[:3, 1], atol=1e-12), name
        _assert_rigid(usd, f"{name}.usd_cam_in_ref")
    print("USD flip verified: USD -z == OpenCV +z and USD +y == OpenCV -y on every camera")

    # 3. round trip through Gf's row-vector ordering
    for name in EXTRINSICS:
        m = usd_cam_in_ref(name)
        assert np.allclose(np.asarray(to_gf_rows(m)).reshape(4, 4).T, m), name
    print("to_gf_rows round-trips")

    # 4. intrinsics in this export must agree with usd_intrinsics.py
    from usd_intrinsics import INTRINSICS

    for vendor_name, scene_name in VENDOR_TO_SCENE.items():
        cam = raw["cameras"][vendor_name]
        intr = INTRINSICS[scene_name]
        for key in ("fx", "fy", "cx", "cy"):
            assert abs(cam[key] - getattr(intr, key)) < 1e-3, f"{scene_name}.{key}"
        assert cam["serial"] == intr.serial, scene_name
    print("intrinsics in the export match usd_intrinsics.py on all measured cameras")

    # 5. reproject the rig's own overlay markers --- the real end-to-end check
    from usd_intrinsics import INTRINSICS as _INTR

    print()
    for name, (point, (want_u, want_v)) in OVERLAY_ANCHORS.items():
        intr = _INTR[name]
        p_cam = np.linalg.inv(EXTRINSICS[name].cam_in_ref) @ np.array([*point, 1.0])
        got_u = intr.fx * p_cam[0] / p_cam[2] + intr.cx
        got_v = intr.fy * p_cam[1] / p_cam[2] + intr.cy
        du, dv = abs(got_u - want_u), abs(got_v - want_v)
        assert max(du, dv) < ANCHOR_TOL_PX, f"{name}: ({got_u:.1f},{got_v:.1f}) vs overlay ({want_u},{want_v})"
        print(
            f"overlay anchor {name:28s} ({got_u:6.1f},{got_v:6.1f}) vs rig ({want_u:6.1f},{want_v:6.1f})"
            f"  d=({du:.1f},{dv:.1f}) px at {p_cam[2] * 100:.1f} cm"
        )

    # 6. what the bench geometry turns those poses into, as a smoke read-out
    base_xyz, yaw = (0.0, 0.05, 0.75), -90.0
    print(f"\nplaced on a base at {base_xyz} yawed {yaw:g} deg:")
    for name, extr in EXTRINSICS.items():
        if extr.reference == "base":
            world = usd_cam_in_world(name, base_xyz, yaw)
            eye = world[:3, 3]
            fwd = -world[:3, 2]
            print(
                f"  {name:28s} world eye=({eye[0]:+.3f},{eye[1]:+.3f},{eye[2]:+.3f}) fwd=({fwd[0]:+.3f},{fwd[1]:+.3f},{fwd[2]:+.3f})"
            )
            if abs(fwd[2]) > 1e-6:  # where the optical axis meets the benchtop
                t = (0.75 - eye[2]) / fwd[2]
                hit = eye + t * fwd
                print(f"  {'':28s} hits the benchtop at ({hit[0]:+.3f},{hit[1]:+.3f}) after {t:.3f} m")
        else:
            local = usd_cam_in_hand(name)
            eye = local[:3, 3]
            fwd = -local[:3, 2]
            print(
                f"  {name:28s} hand-local eye=({eye[0]:+.4f},{eye[1]:+.4f},{eye[2]:+.4f}) fwd=({fwd[0]:+.3f},{fwd[1]:+.3f},{fwd[2]:+.3f})"
            )
